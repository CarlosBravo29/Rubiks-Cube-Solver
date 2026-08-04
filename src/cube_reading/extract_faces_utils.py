from __future__ import annotations
from itertools import combinations
import cv2 as cv
import numpy as np

# Centers of an ideal 3 × 3 grid.
GRID_3X3 = np.array([[col, row] for row in range(3) for col in range(3)], dtype=np.float32)

# Possible positions of a 2 × 2 block inside a 3 × 3 grid.
CELL_POSITIONS = ((0, 0), (1, 0), (0, 1), (1, 1))

# Outer contour of the face (half a cell before and after).
FACE_OUTER_CORNERS = np.array([[-0.5, -0.5], [2.5, -0.5], [2.5, 2.5], [-0.5, 2.5]], dtype=np.float32)

def normalize_centroids(centroids: list | np.ndarray) -> np.ndarray:
    """Converts centroids to a NumPy array (N x 2)."""
    if centroids is None:
        return np.empty((0, 2), dtype=np.float32)
    points = np.asarray(centroids, dtype=np.float32)
    return np.empty((0, 2), dtype=np.float32) if points.size == 0 else points.reshape(-1, 2)

def estimate_sticker_spacing(points: np.ndarray) -> float:
    """Calculates typical sticker spacing based on nearest neighbor distance."""
    if len(points) < 2:
        return 0.0

    nearest_distances = []
    for idx, point in enumerate(points):
        distances = np.linalg.norm(points - point, axis=1)
        distances[idx] = np.inf
        nearest = np.min(distances)
        if np.isfinite(nearest):
            nearest_distances.append(float(nearest))

    return float(np.median(nearest_distances)) if nearest_distances else 0.0

def order_quad_cyclic(quad: np.ndarray) -> np.ndarray:
    """Orders four points cyclically around their center."""
    center = np.mean(quad, axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    return quad[np.argsort(angles)].astype(np.float32)

def generate_quad_orientations(quad: np.ndarray) -> list[np.ndarray]:
    """Generates the 8 possible orientations (rotations in both directions)."""
    orientations = []
    for base_quad in (quad, quad[::-1]):
        for rot in range(4):
            orientations.append(np.roll(base_quad, -rot, axis=0).astype(np.float32))
    return orientations

def canonical_cell_quad(column: int, row: int) -> np.ndarray:
    """Ideal 2 × 2 block inside the 3 × 3 grid."""
    return np.array([
        [column, row],
        [column + 1, row],
        [column + 1, row + 1],
        [column, row + 1]
    ], dtype=np.float32)

def find_local_quads(points: np.ndarray, spacing: float) -> list[np.ndarray]:
    """Finds local groups of four points that form a 2 × 2 block."""
    if len(points) < 4 or spacing <= 0:
        return []

    min_dist, max_dist = spacing * 0.45, spacing * 1.85
    fourth_tol = spacing * 0.75 # tolerance
    detected_quads, used_index_sets = [], set()

    for idx_p, point_p in enumerate(points):
        distances_p = np.linalg.norm(points - point_p, axis=1)
        neighbors = [
            i for i, d in enumerate(distances_p)
            if i != idx_p and min_dist <= d <= max_dist
        ]
        
        for idx_q, idx_r in combinations(neighbors, 2):
            point_q, point_r = points[idx_q], points[idx_r]
            vec_q, vec_r = point_q - point_p, point_r - point_p
            norm_q, norm_r = np.linalg.norm(vec_q), np.linalg.norm(vec_r)

            if norm_q < 1e-6 or norm_r < 1e-6:
                continue

            cosine = np.clip(np.dot(vec_q, vec_r) / (norm_q * norm_r), -1.0, 1.0)
            angle = np.degrees(np.arccos(cosine))

            if not (25.0 <= angle <= 155.0):
                continue

            expected_fourth = point_q + point_r - point_p
            dist_fourth = np.linalg.norm(points - expected_fourth, axis=1)
            idx_s = int(np.argmin(dist_fourth))

            if dist_fourth[idx_s] > fourth_tol:
                continue
            indices = (idx_p, idx_q, idx_r, idx_s)
            if len(set(indices)) != 4:
                continue
            sorted_idx = tuple(sorted(indices))
            if sorted_idx in used_index_sets:
                continue

            quad_points = points[list(sorted_idx)]
            hull = cv.convexHull(quad_points.astype(np.float32)).reshape(-1, 2)

            if len(hull) != 4:
                continue

            area = abs(cv.contourArea(hull.astype(np.float32)))
            if area < spacing**2 * 0.18:
                continue

            detected_quads.append(order_quad_cyclic(hull))
            used_index_sets.add(sorted_idx)

    return detected_quads

def match_projected_grid(projected_grid: np.ndarray, detected_points: np.ndarray, tolerance: float) -> tuple[list[dict], float]:
    """Compares projected grid points against actually detected centroids."""
    possible_matches = []
    for g_idx, proj_p in enumerate(projected_grid):
        distances = np.linalg.norm(detected_points - proj_p, axis=1)
        for d_idx, dist in enumerate(distances):
            if dist <= tolerance:
                possible_matches.append((float(dist), int(g_idx), int(d_idx)))

    possible_matches.sort(key=lambda item: item[0])

    used_grid, used_detected = set(), set()
    matches = []

    for dist, g_idx, d_idx in possible_matches:
        if g_idx in used_grid or d_idx in used_detected:
            continue
        matches.append({"grid_index": g_idx, "detected_index": d_idx, "error": dist})
        used_grid.add(g_idx)
        used_detected.add(d_idx)

    if not matches:
        return [], float("inf")

    mean_error = float(np.mean([m["error"] for m in matches]))
    return matches, mean_error

def get_face_polygon(face: dict) -> np.ndarray:
    """Gets outer corners of a face based on its homography matrix."""
    polygon = cv.perspectiveTransform(FACE_OUTER_CORNERS.reshape(-1, 1, 2), face["homography"]).reshape(-1, 2)
    return polygon.astype(np.float32)

def is_valid_face_polygon(polygon: np.ndarray, image_shape: tuple[int, ...]) -> bool:
    """Rejects degenerate, overly large, or out-of-bounds polygons."""
    if polygon.shape != (4, 2) or not np.all(np.isfinite(polygon)):
        return False

    area = abs(cv.contourArea(polygon.astype(np.float32)))
    img_h, img_w = image_shape[:2]
    img_area = img_h * img_w

    if area < 100 or area > img_area * 0.80:
        return False

    margin_x, margin_y = img_w * 0.25, img_h * 0.25
    if np.any(polygon[:, 0] < -margin_x) or np.any(polygon[:, 0] > img_w + margin_x):
        return False
    if np.any(polygon[:, 1] < -margin_y) or np.any(polygon[:, 1] > img_h + margin_y):
        return False

    return True

def detect_face_candidates(points: np.ndarray, image_shape: tuple[int, ...], min_matches: int = 7) -> list[dict]:
    """Detects all candidate 3 × 3 grids."""
    if len(points) < min_matches:
        return []

    spacing = estimate_sticker_spacing(points)
    if spacing <= 0:
        return []

    local_quads = find_local_quads(points, spacing)
    matching_tol, max_mean_err = spacing * 0.40, spacing * 0.33
    candidates_by_points = {}

    for img_quad in local_quads:
        for orient_quad in generate_quad_orientations(img_quad):
            for col, row in CELL_POSITIONS:
                canon_quad = canonical_cell_quad(col, row)
                homography = cv.getPerspectiveTransform(canon_quad, orient_quad)

                if not np.all(np.isfinite(homography)):
                    continue

                proj_grid = cv.perspectiveTransform(GRID_3X3.reshape(-1, 1, 2), homography).reshape(-1, 2)
                matches, mean_err = match_projected_grid(proj_grid, points, matching_tol)

                if len(matches) < min_matches or mean_err > max_mean_err:
                    continue

                matched_idx = tuple(sorted(m["detected_index"] for m in matches))
                candidate = {"homography": homography, "projected_grid": proj_grid, "matches": matches, "matched_indices": matched_idx, "num_matches": len(matches), "mean_error": mean_err, "spacing": spacing}

                polygon = get_face_polygon(candidate)
                if not is_valid_face_polygon(polygon, image_shape):
                    continue

                candidate["polygon"] = polygon
                candidate["score"] = candidate["num_matches"] * 100 - candidate["mean_error"] * 3

                prev = candidates_by_points.get(matched_idx)
                if prev is None or candidate["score"] > prev["score"]:
                    candidates_by_points[matched_idx] = candidate

    candidates = list(candidates_by_points.values())
    candidates.sort(key=lambda c: (-c["num_matches"], c["mean_error"]))
    return candidates

def select_non_overlapping_faces(candidates: list[dict], maximum_faces: int = 3, maximum_candidates: int = 50) -> list[dict]:
    """Selects up to three faces that do not share centroids."""
    candidates = candidates[:maximum_candidates]
    if not candidates:
        return []

    max_faces = min(maximum_faces, len(candidates))

    for num_faces in range(max_faces, 0, -1):
        best_group, best_score = None, float("-inf")

        for group in combinations(candidates, num_faces):
            point_sets = [set(f["matched_indices"]) for f in group]
            overlap = False

            for i in range(len(point_sets)):
                for j in range(i + 1, len(point_sets)):
                    if point_sets[i] & point_sets[j]:
                        overlap = True
                        break
                if overlap:
                    break
            if overlap:
                continue
            total_score = sum(f["score"] for f in group) + len(set().union(*point_sets)) * 10
            if total_score > best_score:
                best_score = total_score
                best_group = list(group)

        if best_group is not None:
            return best_group
    return []

def rectify_face(image: np.ndarray, face: dict, output_size: int = 300) -> np.ndarray:
    """Transforms the detected face into a square image."""
    src_poly = face["polygon"].astype(np.float32)
    dst_poly = np.array([[0, 0], [output_size - 1, 0], [output_size - 1, output_size - 1], [0, output_size - 1]], dtype=np.float32)
    rect_matrix = cv.getPerspectiveTransform(src_poly, dst_poly)
    return cv.warpPerspective(image, rect_matrix, (output_size, output_size))

def build_face_grid(face: dict) -> list[list[int | None]]:
    """Builds a 3 × 3 matrix containing the centroid indices for each grid cell."""
    grid = [[None for _ in range(3)] for _ in range(3)]
    for match in face["matches"]:
        row, col = match["grid_index"] // 3, match["grid_index"] % 3
        grid[row][col] = match["detected_index"]
    return grid

def draw_face_detection(image: np.ndarray, points: np.ndarray, faces: list[dict]) -> np.ndarray:
    """Draws centroids, polygons, and row/column coordinates on the image."""
    overlay = image.copy()
    face_colors = ((255, 255, 255), (255, 255, 0), (255, 0, 255))
    # Draw all centroids
    for idx, point in enumerate(points):
        x, y = np.round(point).astype(int)
        cv.circle(overlay, (x, y), 5, (0, 255, 0), -1, cv.LINE_AA)
        cv.putText(overlay, str(idx), (x + 5, y - 6), cv.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv.LINE_AA)
    # Draw faces
    for idx, face in enumerate(faces):
        color = face_colors[idx % len(face_colors)]
        polygon = np.round(face["polygon"]).astype(np.int32)
        cv.polylines(overlay, [polygon], True, color, 3, cv.LINE_AA)
        cx, cy = np.round(np.mean(face["polygon"], axis=0)).astype(int)
        cv.putText(overlay, f"Face {idx}: {face['num_matches']}/9", (cx - 50, cy), cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv.LINE_AA)
        for match in face["matches"]:
            row, col = match["grid_index"] // 3, match["grid_index"] % 3
            x, y = np.round(points[match["detected_index"]]).astype(int)
            cv.putText(overlay, f"{row},{col}", (x - 14, y + 17), cv.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv.LINE_AA)
    return overlay

def extract_cube_faces(isolated_cube: np.ndarray, centroids: list | np.ndarray, min_matches: int = 7, maximum_faces: int = 3, output_size: int = 300) -> dict:
    """Main function to detect, rectify, and return Rubik's cube faces."""
    if isolated_cube is None:
        raise ValueError("isolated_cube cannot be None")
    points = normalize_centroids(centroids)
    spacing = estimate_sticker_spacing(points)

    result = {
        "success": False,
        "faces": [],
        "grids": [],
        "rectified_faces": [],
        "overlay": isolated_cube.copy(),
        "points": points,
        "spacing": spacing,
    }

    if len(points) < min_matches:
        return result
    candidates = detect_face_candidates(points, isolated_cube.shape, min_matches)
    faces = select_non_overlapping_faces(candidates, maximum_faces)

    if not faces:
        return result

    result.update({
        "success": True,
        "faces": faces,
        "grids": [build_face_grid(f) for f in faces],
        "rectified_faces": [rectify_face(isolated_cube, f, output_size) for f in faces],
        "overlay": draw_face_detection(isolated_cube, points, faces),
    })
    return result

def print_face_grids(result: dict) -> None:
    """Prints 3 × 3 face matrices to the console."""
    if not result["grids"]:
        print("No faces detected.")
        return

    for idx, grid in enumerate(result["grids"]):
        print(f"\n=== Face {idx} ===")
        for row in grid:
            formatted = [f"{val:3d}" if val is not None else " --" for val in row]
            print(" ".join(formatted))