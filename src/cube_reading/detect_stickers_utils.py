import cv2 as cv
import numpy as np
import math as mt

def get_canny(isolated_cube):
    grey = cv.cvtColor(isolated_cube, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_grey = clahe.apply(grey)
    blur = cv.GaussianBlur(enhanced_grey, (3, 3), 0)
    canny = cv.Canny(blur, 30, 100) #125, 175
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel, iterations=2)

    return canny, closed

def detect_stickers(isolated_cube, closed_img):
    inverted = cv.bitwise_not(closed_img)
    gray = cv.cvtColor(isolated_cube, cv.COLOR_BGR2GRAY)
    _, cube_mask = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
    stickers_only = cv.bitwise_and(inverted, inverted, mask=cube_mask)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    stickers = cv.erode(stickers_only, kernel, iterations=1)
    
    return stickers

def get_sticker_centroids(stickers):
    contours, _ = cv.findContours(stickers, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [], []
    
    raw_areas = [cv.contourArea(cnt) for cnt in contours]
    valid_areas = [a for a in raw_areas if a > 15]

    if not valid_areas:
        return [], []
    
    mid_area = np.median(valid_areas)
    min_area = mid_area * 0.3
    max_area = mid_area * 2.8
    centroids = []
    stickers_contours = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if min_area < area < max_area:
            M = cv.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
                stickers_contours.append(cnt)
    return centroids, stickers_contours

def group_stickers(isolated_cube, centroids):
    output_img = isolated_cube.copy()
    if len(centroids) < 7:
        return output_img
    pts = np.float32(centroids)
    centroids_iter = enumerate(centroids)
    for idx, (x, y) in centroids_iter:
        center = (int(x), int(y))
        cv.circle(output_img, center, 10, (255, 255, 255), -1)
        text = f'{idx}'
        cv.putText(output_img, text, (int(x) - 10, int(y) + 4), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2, cv.LINE_AA)

    return output_img

class StickerDot():
    def __init__(self, idx, coord):
        self.id = idx
        self.coord = coord
        self.vectors = {}
        self.corner_bu = False
        self.t_dot = []
        self.nearest_stickers = []
        self.clock_stickers = []
        self.dimA_stickers = []

    def calc(self, centroids_iter):
        for name, coord in centroids_iter:
            v = ((coord[0] - self.coord[0]), (coord[1] - self.coord[1])) # v = (vx, vy)
            dim = mt.hypot(v[0], v[1])

            angle_rad = mt.atan2(v[1], v[0])

            self.vectors[name] = (v, dim, angle_rad)

    def is_corner(self):
        angles = sorted([data[2] for name, data in self.vectors.items() if name != self.id])
        gaps = [angles[i+1] - angles[i] for i in range(len(angles) - 1)]
        gaps.append(2 * mt.pi - (angles[-1] - angles[0]))
        max_gap = max(gaps)
        span = 2 * mt.pi - max_gap
        self.corner_bu = span < ((59 * mt.pi) / 60) # 177 DEG
        
        return self.corner_bu

    def find_3dots(self):
        sorted_angle = dict(sorted(self.vectors.items(), key=lambda item: item[1][2]))
        sorted_dim = sorted(self.vectors.items(), key=lambda item: item[1][1])
        top_keys = [key for key, val in sorted_dim[:7]]

        low_bound = (179 * mt.pi) / 90 # 358 DEG
        up_bound = (23 * mt.pi) / 45 # 92 DEG

        for name, data in sorted_angle.items():
            if name == self.id:
                continue
            if len(self.t_dot) == 3:
                break
            if ((low_bound <= data[2] <= 2 * mt.pi) or (0 <= data[2] <= up_bound)) and self.corner_bu:
                if name in top_keys:
                    self.t_dot.append(name)

        return self.t_dot

def get_hull_corners(centroids_dict):
        pts = np.array(list(centroids_dict.values()), dtype=np.float32)
        idxs = list(centroids_dict.keys())
        hull = cv.convexHull(pts, returnPoints=False)
        return [idxs[i[0]] for i in hull]

def cluster_angles(angles, gap_threshold=mt.radians(35)):
    """Groups close angles into directional clusters."""
    if not angles:
        return []
    angles = sorted(angles)
    clusters = [[angles[0]]]
    for a in angles[1:]:
        if a - clusters[-1][-1] < gap_threshold:
            clusters[-1].append(a)
        else:
            clusters.append([a])
    if len(clusters) > 1:
        wrap_gap = (2*mt.pi - clusters[-1][-1]) + clusters[0][0]
        if wrap_gap < gap_threshold:
            clusters[0] = clusters[-1] + clusters[0]
            clusters.pop()
    return clusters

def is_junction_vertex(dot, k=6, min_clusters=2, max_clusters=3, balance_ratio=0.35):
    """Detects 'Y' or 'T' junction vertices -- shared across 2-3 faces."""
    sorted_dim = sorted(
        [item for item in dot.vectors.items() if item[0] != dot.id],
        key=lambda item: item[1][1]
    )
    neighbors = sorted_dim[:k]
    angles = [item[1][2] % (2*mt.pi) for item in neighbors]
    clusters = cluster_angles(angles)

    if not (min_clusters <= len(clusters) <= max_clusters):
        return False

    sizes = [len(c) for c in clusters]
    if min(sizes) / max(sizes) < balance_ratio:
        return False

    return True
