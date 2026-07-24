import cv2 as cv
import numpy as np
import detect_cube_utils as cu

def get_canny(frame):
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(grey, (3, 3), cv.BORDER_ISOLATED)
    canny = cv.Canny(blur, 50, 150) # 125, 175 or 50, 150
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv.dilate(canny, kernel, iterations = 2)

    return canny, dilated_edges

def step_2(canny_img):
    contours, _ = cv.findContours(canny_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    blank = cv.cvtColor(np.zeros_like(canny_img), cv.COLOR_GRAY2BGR)
    cnt_lst = []

    for cnt in contours:
        perimeter = cv.arcLength(cnt, closed = True)
        approx = cv.approxPolyDP(cnt, epsilon = perimeter * 0.02, closed = True)        

        M = cv.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv.putText(blank, f'x: {cx}, y: {cy}', (cx, cy), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        cnt_lst.append(cnt)

    if len(cnt_lst) > 0:
        blank = cv.drawContours(blank, cnt_lst, -1, (255, 255, 255), 3)
    return blank

def get_faces(isolated_cube):
    canny, closed = get_canny(isolated_cube)
    tails = step_2(closed)
    return tails