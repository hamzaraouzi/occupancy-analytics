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
        self.line = [(100, 200), (500, 200)] #TODO: to be changed later.
        self.msg_broker = None

        if not self.cap.isOpened():
            logger.error("Error: Unable to open RTSP stream.")
            exit()

        self.obj_history = dict()

    def process_tracks(self, results, frame):
        for result in results:
            for box in result.boxes:
                object_id = int(box.id.numpy()[0])
                bbox = box.xyxy[0].numpy().tolist()

                center = calculate_center(bbox=bbox)
                prev_center = self.obj_history.get(object_id, None)
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

            results = self.model.track(source=frame, tracker="bytetrack.yaml")
            self.process_tracks(results=results, frame=frame)