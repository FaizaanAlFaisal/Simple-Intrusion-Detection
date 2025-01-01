import cv2
import numpy as np
from ultralytics import YOLO
from utils import VideoCapture


def centroid_near_line(centroid_x:float, centroid_y:float, line_point1:tuple, line_point2:tuple, threshold:float=10) -> bool:
    """
    Determines if a centroid has crossed or is near a line defined by two points.

    Parameters:
    centroid_x (float): The x-coordinate of the centroid.
    centroid_y (float): The y-coordinate of the centroid.
    line_point1 (tuple): A tuple representing the first point of the line (x1, y1).
    line_point2 (tuple): A tuple representing the second point of the line (x2, y2).
    threshold (float): The distance threshold within which the centroid is considered near the line.

    Returns:
    bool: True if the centroid is near or has crossed the line, False otherwise.
    """
    # coords for line points
    x1, y1 = line_point1
    x2, y2 = line_point2

    # direction vector
    line_vector = np.array([x2 - x1, y2 - y1])

    # vector from the first line point to the centroid
    centroid_vector = np.array([centroid_x - x1, centroid_y - y1])

    # perp dist from centroid to line
    cross_product = np.abs(np.cross(line_vector, centroid_vector))  
    line_length = np.linalg.norm(line_vector)
    distance = cross_product / line_length

    return distance <= threshold


yolo = YOLO("model/yolov8m.pt")

cap = VideoCapture("data/vid2.mp4", capped_fps=True, restart_on_end=True, framerate=20)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

cap.start()

cv2lines = [(950,700),(950,20)]
# cv2lines = [(20, 200),(1000,20)]
# cv2lines = [(700,500),(300,150)]


while True:
    ret, frame = cap.read()

    if not ret:
        # print("Error: Failed to capture frame.")
        continue

    annotated_frame = cv2.resize(frame, (1280, 720))

    # yolo detection - people only
    results = yolo.track(annotated_frame, classes=[0], verbose=False, stream=True, persist=True)

    for res in results:
        if res.boxes.id is None:
            continue

        id = res.boxes.id.int().cpu().tolist()
        
        for detection in res.boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if centroid_near_line(cx, cy, cv2lines[0], cv2lines[1], threshold=100):
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # annotated_frame = res.plot()

    annotated_frame = cv2.line(annotated_frame, cv2lines[0], cv2lines[1], (0, 0, 255), 2)

    # show the frame
    cv2.imshow('Intrusion Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()