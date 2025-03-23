import cv2
import logging
from utils import (calculate_center, has_crossed_line,
                   update_obj_history)

from queue import Queue
from yolov8_tensorrt import YOLOv8TensorRT
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LineCrossing(threading.Thread):
    def __init__(self, source, model, tracker, line, msg_queue: Queue):
        super().__init__()
        self.model = YOLOv8TensorRT(engine_path=model)
        self.source = source
        self.line = line
        self.msg_queue = msg_queue
        self.tracker = tracker
        self.obj_history = dict()
        self.runnig = False

    def process_tracks(self, tracked_ojects, frame):
        for obj in tracked_ojects:
            object_id = obj["object_id"]
            bbox = obj["bbox"]

            center = calculate_center(bbox=bbox)
            prev_center = self.obj_history.get(object_id, None)
            line_is_crossed, direction = has_crossed_line(prev_center,
                                                          center,
                                                          self.line)
            if line_is_crossed:
                logging.info(f"Object {object_id}: {direction} ")
                self.msg_queue.put({"direction": direction})

                self.obj_history = update_obj_history(
                    object_histories=self.obj_history,
                    object_id=object_id,
                    center=center)

    def run(self):
        self.runnig = True
        cap = cv2.VideoCapture(self.source)
        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                logger.error("Stream ended or error occured")
                break

            bboxes, scores, class_ids = self.model.infer(frame=frame)
            tracked_objects = self.tracker.track_objects(bboxes, scores, (720, 1280))
            self.process_tracks(tracked_objects)

    def stop(self):
        self.running = False