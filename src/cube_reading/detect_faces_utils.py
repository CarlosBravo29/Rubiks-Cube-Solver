import cv2 as cv
import numpy as np
import math as mt
import detect_cube_utils as cu

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

def get_stiker_centroids(stikers):
    contours, _ = cv.findContours(stikers, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

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
    stikers_contours = []

    for cnt in contours:
        area = cv.contourArea(cnt)

        if min_area < area < max_area:
            M = cv.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
                stikers_contours.append(cnt)

    return centroids, stikers_contours

def group_stikers(isolated_cube, centroids):
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


class Sticker_dot():
    def __init__(self, idx, coord):
        self.id = idx
        self.coord = coord
        self.vectors = {}
        self.corner_bu = False
        self.nearest_stickers = []

    def calc(self, centroids_iter):
        for name, coord in centroids_iter:
            v = ((coord[0] - self.coord[0]), (coord[1] - self.coord[1])) # v = (vx, vy)
            dim = mt.hypot(v[0], v[1])
            is_right = v[0] > 0
            is_left  = v[0] < 0
            is_above = v[1] > 0
            is_below = v[1] < 0
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
            if len(self.nearest_stickers) == 3:
                break
            if ((low_bound <= data[2] <= 2 * mt.pi) or (0 <= data[2] <= up_bound)) and self.corner_bu:
                if name in top_keys:
                    self.nearest_stickers.append(name)

        return self.nearest_stickers


### Testing sticker grouping to detect faces by finding the maximum angular gap between points. (Works! Correctly detects corners)

def test_group(my_coords = [(0,(12.25,3)),(1,(9,3)),(2,(15,5)),(4,(12.5,5.5)),(5,(6,5)),(6,(9,6)),(7,(18,7)),(8,(3,8)),(9,(15.5,8)),(10,(6,8)),(11,(13,8.5)),(12,(9.5,9)),(13,(18.5,10)),(14,(3,11)),(15,(16.25,11)),(16,(6,11)),(17,(11.5,11.5)),(18,(19.25,13.25)),(19,(14.5,13.5)),(20,(8,14)),(21,(3,14)),(22,(12,15.5)),(23,(17.5,15.5)),(24,(14,18)),(25,(11,20))]):
    my_stickers = []
    for name, coord in my_coords:
        sticker = Sticker_dot(name, coord)
        sticker.calc(my_coords)
        my_stickers.append(sticker)
        print(f"{sticker.id}: {sticker.is_corner()},   {sticker.find_3dots()}")
    return my_stickers

a = [(0,(12.5,2.5)),(1,(7.5,3.5)),(2,(13.75,4)),(3,(3.5,4.5)),(4,(8.5,5.5)),(5,(15.25,6)),(6,(4,6.25)),(7,(9.5,7.5)),(8,(4.5,8.5)),(9,(16.25,9.5)),(10,(10.5,10.5)),(11,(5.25,11.5)),(12,(16.5,13.5)),(13,(11,14.5)),(14,(5.75,15.5)),(15,(16.25,17.5)),(16,(11.5,18.5)),(17,(6.5,19))]
test_group(a)

def try_grid(stickers_lst):
    for item in stickers_lst:
        pass 