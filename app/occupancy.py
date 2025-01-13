import cv2
from ultralytics import YOLO
import logging
from app.utils import (calculate_center, has_crossed_line, prepare_osd_frames,
                       update_obj_history)

from queue import Queue
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Occupancy:
    def __init__(self, source, model, osd_queue: Queue, msg_queue: Queue):
        self.model = YOLO(model).to("cuda")
        self.cap = cv2.VideoCapture(source)
        self.line = [(750, 200), (950, 1250)]
        self.msg_queue = msg_queue
        self.osd_queue = osd_queue

        if not self.cap.isOpened():
            logger.error("Error: Unable to open RTSP stream.")
            exit()

        self.obj_history = dict()

    def process_tracks(self, results, frame):
        for result in results:
            for box in result.boxes:
                object_id = int(box.id.cpu().numpy()[0])
                bbox = box.xyxy[0].cpu().numpy().tolist()
                cls = int(box.cls.cpu())
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

                osd_frame = prepare_osd_frames(frame=frame, bbox=bbox,
                                               center=center,
                                               line=self.line,
                                               obj_id=object_id)

            self.osd_queue.put(osd_frame)

    def run(self):
        frame_skip_cnt = 0
        while self.cap.isOpened():
            if frame_skip_cnt % 5 != 0:
                continue

            ret, frame = self.cap.read()
            if not ret:
                logger.error("Stream ended or error occured")
                break

            results = self.model.track(source=frame, persist=True,
                                       tracker="bytetrack.yaml")
            self.process_tracks(results=results, frame=frame)
    
        # ßwrite_output_video(self.osd_buffer, "out.mp4", fps=25)