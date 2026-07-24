import cv2 as cv
import detect_cube_utils as dc
import detect_faces_utils as fc

width_frame = 600
height_frame = 440

cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_FRAME_WIDTH, width_frame)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height_frame)

while True:
    isTrue, frame = cap.read()

    isolated_cube = dc.find_cube(frame)
    tails = fc.get_faces(isolated_cube)

    cv.imshow('Original', frame)
    cv.imshow('Isolated cube', isolated_cube)

    #test
    canny, dil = fc.get_canny(isolated_cube)
    cv.imshow('Canny', canny)
    cv.imshow('dilated', dil)
    cv.imshow('Tails', tails)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

cap.release()
cv.destroyAllWindows()