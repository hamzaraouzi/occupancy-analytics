import cv2
from ultralytics import YOLO
import logging
from utils import (calculate_center, has_crossed_line, prepare_osd_frames,
                   update_obj_history, write_output_video)
from msghandler import MessageHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Occupancy:
    def __init__(self, source, model):
        self.model = YOLO(model).to("cuda")
        self.cap = cv2.VideoCapture(source)
        self.line = [(750, 200), (950, 1250)] #TODO: to be changed later.
        self.msg_handler = MessageHandler()

        if not self.cap.isOpened():
            logger.error("Error: Unable to open RTSP stream.")
            exit()

        self.obj_history = dict()
        self.osd_buffer = list()

    def process_tracks(self, results, frame):
        for result in results:
            for box in result.boxes:
                object_id = int(box.id.cpu().numpy()[0])
                bbox = box.xyxy[0].cpu().numpy().tolist()
                print(bbox, "+++++++++++++++++")
                cls = None #TODO read the class
                center = calculate_center(bbox=bbox)
                prev_center = self.obj_history.get(object_id, None)
                line_is_crossed, direction = has_crossed_line(prev_center, center, self.line)
                if line_is_crossed:
                    logging.info(f"Object {object_id}: {direction} ")
                    #self.msg_handler.send_event(direction, cls)

                self.obj_history = update_obj_history(
                    object_histories=self.obj_history,
                    object_id=object_id,
                    center=center)

                osd_frame = prepare_osd_frames(frame=frame, bbox=bbox,
                                               center=center,
                                               line=self.line,
                                               obj_id=object_id)

                self.osd_buffer.append(osd_frame)

    def run(self):
        frame_skip_cnt = 0
        while self.cap.isOpened():
            if frame_skip_cnt % 5 != 0:
                continue

            ret, frame = self.cap.read()
            if not ret:
                logger.error("Stream ended or error occured")
                break

            results = self.model.track(source=frame, persist=True, tracker="bytetrack.yaml")
            self.process_tracks(results=results, frame=frame)
        
        write_output_video(self.osd_buffer, "out.mp4", fps=25)