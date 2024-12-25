import cv2
from ultralytics import YOLO
import  logging 
from utils import calculate_center, has_crossed_line, prepare_osd_frames, update_obj_history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Occupancy:
    def __init__(self, source, model):
        self.model = YOLO(model)
        self.cap = cv2.VideoCapture(source)
        self.line = "..."
        self.msg_broker = None

        if not self.cap.isOpened():
            logger.error("Error: Unable to open RTSP stream.")
            exit()

        self.obj_history = dict()

    def process_tracks(self, results, frame):
        for track in results.boxes.tracks:
            object_id = track.id
            bbox = track.xyxy
            center = calculate_center(bbox=bbox)
            prev_center = self.prev_center[object_id]
            if has_crossed_line(prev_center, center, self.line):
                logger.info(f"Object {object_id} crossed the line!")

            self.obj_history = update_obj_history(
                object_histories=self.obj_history,
                object_id=object_id,
                center=center)

            prepare_osd_frames(frame=frame, bbox=bbox,
                               center=center,
                               line=self.line)

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Stream ended or error occured")
                break

            results = self.model.track(source=frame, stream=True, tracker="bytetrack.yaml")
            self.process_tracks(results=results, frame=frame)