import cv2 as cv
import detect_cube_utils as dc
import detect_stickers_utils as ds
import extract_faces_utils as fe
import face_color_extraction_utils as ce


width_frame = 600
height_frame = 440

detected_face_colors = set()

cap = cv.VideoCapture(1)
cap.set(cv.CAP_PROP_FRAME_WIDTH, width_frame)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height_frame)

while True:
    isTrue, frame = cap.read()

    isolated_cube = dc.find_cube(frame)
    if isolated_cube is None:
        continue

    canny, closed = ds.get_canny(isolated_cube)
    stickers = ds.detect_stickers(isolated_cube, closed)
    centroids, contours = ds.get_sticker_centroids(stickers)

    if centroids is None or len(centroids) == 0:
        continue

    coords_fmt = [(index, (round(float(x), 2), round(float(y), 2))) for index, (x, y) in enumerate(centroids)]

    face_result = fe.extract_cube_faces(
        isolated_cube=isolated_cube,
        centroids=centroids,
        min_matches=8,
        maximum_faces=3,
        output_size=300,
    )

    for face_index, rectified_face in enumerate(face_result["rectified_faces"]):
        if not ce.valid_face(rectified_face):
            continue
        face_color = ce.is_already_read(rectified_face)
        if face_color != "Un" and face_color not in detected_face_colors:
            detected_face_colors.add(face_color)
            print(f"Nueva cara detectada: {face_color}")
            window_name = f"Cara Detectada - {face_color}"
            cv.imshow(window_name, rectified_face)
    
    cv.imshow("Stickers mask", stickers)
    cv.imshow("Isolated", isolated_cube)
    cv.imshow("Face detection", face_result["overlay"])
    cv.imshow("Original", frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

cap.release()
cv.destroyAllWindows()