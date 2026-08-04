import cv2 as cv
import detect_cube_utils as dc
import detect_stickers_utils as ds

width_frame = 600
height_frame = 440

cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_FRAME_WIDTH, width_frame)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height_frame)

while True:
    isTrue, frame = cap.read()

    ### TEST CUBE DETECTION ###
    # isolated_cube = dc.find_cube(frame)
    # canny, closed = dc.get_canny(frame)

    # cv.imshow('Original', frame)
    # cv.imshow('Isolated cube', isolated_cube)
    # cv.imshow('Canny', canny)
    # cv.imshow('Closed', closed)
    ### END TEST CUBE DETECTION ###

    ### TEST FACES DETECTION ###
    isolated_cube = dc.find_cube(frame)

    #border = dS.get_boundary(isolated_cube)
    #_, mask = ds.get_inner_lines(border)
    canny, closed = ds.get_canny(isolated_cube)
    stikers = ds.detect_stickers(isolated_cube, closed)
    centroids, _ = ds.get_stiker_centroids(stikers)
    oi = ds.group_stikers(isolated_cube, centroids)
    
    cv.imshow('Isolated_cube', isolated_cube)
    cv.imshow('Closed', closed)
    cv.imshow('SP', stikers)
    cv.imshow('oi', oi)

    #cv.imshow('mask', mask)
    ### END TEST FACES DETECTION ###

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

cap.release()
cv.destroyAllWindows()