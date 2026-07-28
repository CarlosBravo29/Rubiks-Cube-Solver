import cv2 as cv
import numpy as np
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

def grup_stikers(isolated_cube, centroids):
    output_img = isolated_cube.copy()
    if len(centroids) < 7:
        return output_img
    pts = np.float32(centroids)

    #x, y, w, h = cv.boundingRect(points)

    for idx, (x, y) in enumerate(centroids):
        center = (int(x), int(y))

        cv.circle(output_img, center, 10, (255, 255, 255), -1)
        text = f'{idx}'
        cv.putText(output_img, text, (int(x) - 10, int(y) + 4), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2, cv.LINE_AA)

    return output_img
