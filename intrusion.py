import cv2
import numpy as np
from ultralytics import YOLO
from utils import VideoCapture
import threading


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


class IntrusionDetection:
    """
    class IntrusionDetection
    ------------------------
    This class is designed to detect intrusions based on object detection using YOLOv5.

    The primary function is to detect objects in a video feed and determine if any objects cross a line/defined threshold.

    Parameters
    ----------
        video_source (str): Path to a video file or a video feed URL (rtsp/rtmp/http).
        lines (tuple): A tuple of two points defining a line (x1, y1), (x2, y2).
        model_path (str): Path to the YOLOv5 model file.
        intrusion_threshold (int): The distance threshold within which an object is considered to have crossed the line.
        intrusion_flag_duration (int): The number of frames to keep the intrusion flag active after an intrusion is detected.
        capped_fps (bool): If True, caps the frame rate. Set it to true for file playback.
        restart_on_end (bool): If True, restarts video file playback when it ends.
        framerate (int): Frame rate for video file playback (used if capped_fps is True).
        crop (tuple): A tuple of two points defining a crop region (x1, y1), (x2, y2).
        resize (tuple): A tuple of two values defining the resize dimensions (width, height).
    """

    def __init__(
                    self, video_source:str, lines:tuple, show_line:bool=False, model_path:str="model/yolov8m.pt",
                    intrusion_threshold:int=120, intrusion_flag_duration:int=15, capped_fps:bool=True, restart_on_end:bool=True, 
                    framerate:int=20, crop:tuple=None, resize:tuple=(1280, 720),
                ):
        # model and video cap setup
        self.yolo = YOLO(model_path)
        self.cap = VideoCapture(video_source, capped_fps=capped_fps, restart_on_end=restart_on_end, framerate=framerate)
        if not self.cap.isOpened():
            print("Error: Could not access the feed.")
            exit()

        # intrusion detection parameters
        self.show_line = show_line
        self.lines = lines
        self.intrusion_threshold = intrusion_threshold
        self.intrusion_flag_duration = intrusion_flag_duration
        self.intrusion_flag = False

        # crop and resize setup
        self.crop = crop
        self.resize = resize

        # threading setup
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.__intrusion_detection_thread__)
        self.thread.daemon = True
        self.thread.start()


    def __intrusion_detection_thread__(self):
        """
        Continuously reads frames from the video source in a separate thread.
        """

        self.cap.start()
        flag_frame_count = self.intrusion_flag_duration

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            if self.crop is not None:
                annotated_frame = frame[self.crop[0][1]:self.crop[1][1], self.crop[0][0]:self.crop[1][0]]
            else:
                annotated_frame = frame

            annotated_frame = cv2.resize(annotated_frame, self.resize)

            results = self.yolo.track(annotated_frame, classes=[0], verbose=False, stream=True, persist=True)

            for res in results:
                if res.boxes.id is None:
                    continue

                for detection in res.boxes:
                    x1, y1, x2, y2 = map(int, detection.xyxy[0])
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    if centroid_near_line(cx, cy, self.lines[0], self.lines[1], threshold=self.intrusion_threshold):
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        self.intrusion_flag = True
                    else:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if self.intrusion_flag:
                cv2.putText(annotated_frame, f"INTRUSION DETECTED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                flag_frame_count -= 1
                if flag_frame_count == 0:
                    self.intrusion_flag = False
                    flag_frame_count = self.intrusion_flag_duration
            
            if self.show_line:
                annotated_frame = cv2.line(annotated_frame, self.lines[0], self.lines[1], (0, 0, 255), 2)
            
            cv2.imshow('Intrusion Detection', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def stop(self):
        """
        Stops the intrusion detection process.
        """
        self.stop_event.set()
        self.cap.release()
        self.thread.join(2)


# driver code
def main():
    
    # cv2lines = [(950,700),(950,20)]
    cv2lines = [(20,450),(1000,250)]
    # cv2lines = [(300,200),(600,300)]

    intr_det = IntrusionDetection(  
                                    "data/vid4.mp4", cv2lines, show_line=True, capped_fps=True, 
                                    restart_on_end=True, framerate=20, intrusion_threshold=120, 
                                    intrusion_flag_duration=15, resize=(1280, 720), crop=((0,0), (1280,720))
                                )
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            intr_det.stop()
            break


if __name__ == "__main__":
    main()