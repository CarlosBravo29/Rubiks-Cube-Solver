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
    return np.array([[column, row], [column + 1, row], [column + 1, row + 1], [column, row + 1]], dtype=np.float32)

def find_local_quads(points: np.ndarray, spacing: float) -> list[np.ndarray]:
    """Finds local groups of four points that form a 2 × 2 block."""
    if len(points) < 4 or spacing <= 0:
        return []

    min_dist, max_dist = spacing * 0.45, spacing * 1.85
    fourth_tol = spacing * 0.75 # tolerance
    detected_quads, used_index_sets = [], set()

    for idx_p, point_p in enumerate(points):
        distances_p = np.linalg.norm(points - point_p, axis=1)
        neighbors = [i for i, d in enumerate(distances_p) if i != idx_p and min_dist <= d <= max_dist]

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
    if len(projected_grid) == 0 or len(detected_points) == 0:
        return [], float("inf")
    differences = (projected_grid[:, np.newaxis, :] - detected_points[np.newaxis, :, :])
    distance_matrix = np.linalg.norm(differences, axis=2)
    grid_indices, detected_indices = np.where(distance_matrix <= tolerance)

    if len(grid_indices) == 0:
        return [], float("inf")

    distances = distance_matrix[grid_indices, detected_indices]
    order = np.argsort(distances)

    used_grid = np.zeros(len(projected_grid), dtype=bool)
    used_detected = np.zeros(len(detected_points), dtype=bool)

    matches = []

    for position in order:
        g_idx = int(grid_indices[position])
        d_idx = int(detected_indices[position])

        if used_grid[g_idx] or used_detected[d_idx]:
            continue

        distance = float(distances[position])
        matches.append({"grid_index": g_idx, "detected_index": d_idx, "error": distance})
        used_grid[g_idx] = True
        used_detected[d_idx] = True

    if not matches:
        return [], float("inf")

    mean_error = float(np.mean([match["error"] for match in matches]))

    return matches, mean_error

def get_face_polygon(face: dict) -> np.ndarray:
    """Gets outer corners of a face based on its homography matrix."""
    polygon = cv.perspectiveTransform(FACE_OUTER_CORNERS.reshape(-1, 1, 2), face["homography"]).reshape(-1, 2)
    return polygon.astype(np.float32)

def polygon_shape_regularity(polygon: np.ndarray, max_side_ratio: float = 1.9, min_angle: float = 50.0, max_angle: float = 130.0) -> bool:
    """Rejects severely skewed or deformed quadrilaterals based on side ratios and internal angles."""
    if polygon.shape != (4, 2):
        return False
    pts = polygon.astype(np.float64)
    sides = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]

    if min(sides) < 1e-6:
        return False
    if max(sides) / min(sides) > max_side_ratio:
        return False

    for i in range(4):
        prev_pt = pts[(i - 1) % 4]
        curr_pt = pts[i]
        next_pt = pts[(i + 1) % 4]
        v1 = prev_pt - curr_pt
        v2 = next_pt - curr_pt
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return False
        cosine = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine))
        if not (min_angle <= angle <= max_angle):
            return False
    return True

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
    if not polygon_shape_regularity(polygon):
        return False
    return True

def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    norm_1 = np.linalg.norm(v1)
    norm_2 = np.linalg.norm(v2)

    if norm_1 < 1e-6 or norm_2 < 1e-6:
        return 180.0
    cosine = np.clip(np.dot(v1, v2) / (norm_1 * norm_2), -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))

def candidate_grid_points(candidate: dict, detected_points: np.ndarray) -> np.ndarray | None:
    """Organizes candidate points into a 3x3 grid matrix, returning None if incomplete."""
    grid = np.full((3, 3, 2), np.nan, dtype=np.float32)

    for match in candidate["matches"]:
        grid_index = match["grid_index"]
        detected_index = match["detected_index"]
        row = grid_index // 3
        col = grid_index % 3
        grid[row, col] = detected_points[detected_index]

    if np.isnan(grid).any():
        return None
    return grid

def grid_direction_consistency(candidate: dict, detected_points: np.ndarray, max_direction_change: float = 22.0, max_spacing_ratio: float = 2.0) -> bool:
    """Validates grid alignment by enforcing directional and spacing consistency across rows and columns."""
    grid = candidate_grid_points(candidate, detected_points)

    if grid is None:
        return False

    horizontal_vectors = []
    vertical_vectors = []

    for row in range(3):
        v_left = grid[row, 1] - grid[row, 0]
        v_right = grid[row, 2] - grid[row, 1]
        direction_change = angle_between_vectors(v_left, v_right)

        if direction_change > max_direction_change:
            return False

        horizontal_vectors.extend([v_left, v_right])

    for col in range(3):
        v_top = grid[1, col] - grid[0, col]
        v_bottom = grid[2, col] - grid[1, col]
        direction_change = angle_between_vectors(v_top, v_bottom)
        if direction_change > max_direction_change:
            return False
        vertical_vectors.extend([v_top, v_bottom])

    horizontal_lengths = np.array([np.linalg.norm(vector) for vector in horizontal_vectors], dtype=np.float32)
    vertical_lengths = np.array([np.linalg.norm(vector) for vector in vertical_vectors], dtype=np.float32)

    if np.min(horizontal_lengths) < 1e-6:
        return False

    if np.min(vertical_lengths) < 1e-6:
        return False

    horizontal_ratio = (np.max(horizontal_lengths) / np.min(horizontal_lengths))
    vertical_ratio = (np.max(vertical_lengths) / np.min(vertical_lengths))

    if horizontal_ratio > max_spacing_ratio:
        return False
    if vertical_ratio > max_spacing_ratio:
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
    matching_tol, max_mean_err = spacing * 0.27, spacing * 0.18
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

                if not grid_direction_consistency(candidate, points, max_direction_change=22.0, max_spacing_ratio=2.0):
                    continue
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