import cv2 as cv
import numpy as np

def get_canny(frame):
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(grey, (5, 5), 0)
    canny = cv.Canny(blur, 30, 100) #125, 175
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
    closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel, iterations=2)

    return canny, closed

def cube_mask(closed_img, min_area=5000):
    contours, _ = cv.findContours(closed_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(closed_img)
    cnt_lst = []

    valid_contours = []
    for cnt in contours:
        area = cv.contourArea(cnt)

        if area < min_area:
            continue

        x, y, w, h = cv.boundingRect(cnt)
        asperct_ratio = float(w) / h

        if 0.7 <= asperct_ratio <= 1.3:
            fill_percentage = float(area) / (w * h)
            if fill_percentage > 0.40:
                hull = cv.convexHull(cnt)
                valid_contours.append(hull)
                # test
                #cv.rectangle(closed_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                #cv.imshow('Test', closed_img)
                # end test

    if valid_contours:
        cv.drawContours(mask, valid_contours, -1, 255, thickness=cv.FILLED)        
    return mask

def isolate_cube(frame, mask):
    cube = cv.bitwise_and(frame, frame, mask=mask)
    pixels = cube[mask > 0]
    data = np.float32(pixels)
    return data, cube

def find_cube(frame):
    canny, closed = get_canny(frame)
    mask = cube_mask(closed)
    _, cube = isolate_cube(frame, mask)
    return cube