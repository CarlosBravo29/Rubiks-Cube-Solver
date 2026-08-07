import cv2 as cv
import numpy as np

def classify_pixels(hsv: np.ndarray) -> np.ndarray:
    """Classifies each pixel in an HSV image by color with white/dark priority (W > Un > R > O > Y > G > B)."""
    h = hsv[..., 0].astype(np.float32)
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)
    labels = np.full(h.shape, "", dtype=object)
    undecided = np.ones(h.shape, dtype=bool)

    def assign(mask, code):
        nonlocal undecided
        apply_ = mask & undecided
        labels[apply_] = code
        undecided[apply_] = False

    assign((s < 50) & (v > 150), "W")
    assign((s < 60) & (v < 100), "Un")
    assign((h <= 6) | (h >= 175), "R")
    assign((h >= 7) & (h <= 20), "O")
    assign((h >= 21) & (h <= 34), "Y")
    assign((h >= 35) & (h <= 85), "G")
    assign((h >= 86) & (h <= 130), "B")
    labels[undecided] = "Un"

    return labels

def majority_color(roi_bgr: np.ndarray, exclude_unknown: bool = True, debug: bool = False) -> str:
    """Classifies a BGR region of interest by returning the majority color, optionally ignoring unknown ('Un') pixels."""
    hsv = cv.cvtColor(roi_bgr, cv.COLOR_BGR2HSV)
    labels = classify_pixels(hsv)
    flat = labels.flatten()
    vals, counts = np.unique(flat, return_counts=True)

    if debug:
        for val, cnt in sorted(zip(vals, counts), key=lambda x: -x[1]):
            print(f"    {val}: {cnt}")
    if exclude_unknown:
        mask = vals != "Un"
        if mask.any():
            vals, counts = vals[mask], counts[mask]

    return str(vals[np.argmax(counts)])

def is_already_read(face_frame: np.ndarray, debug: bool = False):
    """Takes a ROI of a face and returns its color code (center sticker)."""
    h, w = face_frame.shape[:2]
    dx, dy = w // 3, h // 3
    cx, cy = w // 2, h // 2
    x1, x2 = cx - (dx // 2), cx + (dx // 2)
    y1, y2 = cy - (dy // 2), cy + (dy // 2)
    roi = face_frame[y1:y2, x1:x2]

    color = majority_color(roi, debug=debug)
    if debug:
        print(f"[color] -> {color}")
    return color

def _edge_black_ratios(mask: np.ndarray, strip_frac: float = 0.12) -> dict:
    """Calculates the ratio of black pixels along each edge of the mask."""
    h, w = mask.shape
    sy, sx = max(1, int(h * strip_frac)), max(1, int(w * strip_frac))
    edges = {"top":    mask[0:sy, :], "bottom": mask[h - sy:h, :], "left":   mask[:, 0:sx], "right":  mask[:, w - sx:w]}
    return {k: np.count_nonzero(v == 0) / v.size for k, v in edges.items()}

def sharpness_score(face_frame: np.ndarray) -> float:
    """Calculates image sharpness using the variance of the Laplacian operator."""
    gray = cv.cvtColor(face_frame, cv.COLOR_BGR2GRAY)
    return cv.Laplacian(gray, cv.CV_64F).var()

def valid_face(face_frame: np.ndarray, max_black_ratio: float = 0.18, max_edge_black_ratio: float = 0.16, min_sharpness: float = 40.0, debug: bool = False) -> bool:
    """Evaluates if a face crop is valid based on black pixel ratio, edge alignment, and image sharpness."""
    h, w = face_frame.shape[:2]
    gray = cv.cvtColor(face_frame, cv.COLOR_BGR2GRAY)
    _, dark_mask = cv.threshold(gray, 30, 255, cv.THRESH_BINARY_INV)
    black_ratio = np.count_nonzero(dark_mask) / (h * w)
    if black_ratio > max_black_ratio:
        if debug:
            print(f"[valid_face] RECHAZADA: black_ratio total={black_ratio:.3f}")
        return False

    bg_mask = cv.bitwise_not(dark_mask)
    edge_ratios = _edge_black_ratios(bg_mask)
    worst_edge, worst_val = max(edge_ratios.items(), key=lambda kv: kv[1])
    if worst_val > max_edge_black_ratio:
        if debug:
            print(f"[valid_face] RECHAZADA: borde '{worst_edge}'={worst_val:.3f} "
                  f"(todos: {edge_ratios})")
        return False

    sharp = sharpness_score(face_frame)
    if sharp < min_sharpness:
        if debug:
            print(f"[valid_face] RECHAZADA: nitidez={sharp:.1f} < {min_sharpness}")
        return False

    if debug:
        print(f"[valid_face] OK: total={black_ratio:.3f} bordes={edge_ratios} "
              f"nitidez={sharp:.1f}")
    return True

def debug_hue_calibration(face_frame: np.ndarray, label: str = ""):
    """Prints the majority color for each grid cell in a 3x3 face frame for calibration."""
    h, w = face_frame.shape[:2]
    cell_h, cell_w = h // 3, w // 3
    inset = 0.9
    roi_w, roi_h = int(cell_w * inset), int(cell_h * inset)
    print(f"\n=== Calibracion {label} ===")
    for row in range(3):
        line = []
        for col in range(3):
            cy, cx = row * cell_h + cell_h // 2, col * cell_w + cell_w // 2
            roi = face_frame[cy - roi_h // 2:cy + roi_h // 2,
                              cx - roi_w // 2:cx + roi_w // 2]
            line.append(majority_color(roi))
        print("  " + " | ".join(line))