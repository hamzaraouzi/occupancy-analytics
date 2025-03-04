import cv2
from ultralytics import YOLO
import logging
from utils import (calculate_center, has_crossed_line,
                   update_obj_history)

from queue import Queue
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Occupancy:
    def __init__(self, source, model, tracker, line, msg_queue: Queue):
        self.model = model
        self.cap = cv2.VideoCapture(source)
        self.line = line
        self.msg_queue = msg_queue
        self.tracker = tracker

        if not self.cap.isOpened():
            logger.error("Error: Unable to open RTSP stream.")
            exit()

        self.obj_history = dict()

    def process_tracks(self, tracks, frame):
        for track in tracks:
            object_id = track.track_id
            bbox = track.tlbr
            cls = int(track.class_id)

            center = calculate_center(bbox=bbox)
            prev_center = self.obj_history.get(object_id, None)
            line_is_crossed, direction = has_crossed_line(prev_center,
                                                          center,
                                                          self.line)
            if line_is_crossed:
                logging.info(f"Object {object_id}: {direction} ")
                self.msg_queue.put({"classe": cls,
                                    "direction": direction})

                self.obj_history = update_obj_history(
                    object_histories=self.obj_history,
                    object_id=object_id,
                    center=center)

    def run(self):

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_skip_cnt = 0

        while self.cap.isOpened():
            if frame_skip_cnt % 3 != 0:
                continue

            ret, frame = self.cap.read()
            if not ret:
                logger.error("Stream ended or error occured")
                break

            detections = self.model.infer(frame)
            tracks = self.tracker.update(detections, [height, width],
                                         (height, width))
            self.process_tracks(tracks=tracks, frame=frame)
            frame_skip_cnt+=1
        # write_output_video(self.osd_buffer, "out.mp4", fps=25)