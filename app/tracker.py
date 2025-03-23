import numpy as np
from yolox.tracker.byte_tracker import BYTETracker
import argparse

class ObjectTracker:
    def __init__(self. **kwargs):
        args = self.make_parser().parse_args()
        self.tracker = BYTETracker(args)
    
    def make_parser(self):
        parser = argparse.ArgumentParser("ByteTracker arguments")

        parser.add_argument("--track_thresh", type=float, default=0.5,
                            help="tracking confidence threshold")
        parser.add_argument("--track_buffer", type=int, default=30,
                            help="the frames for keep lost tracks")
        parser.add_argument("--match_thresh", type=float, default=0.6,
                            help="matching threshold for tracking")
        
        parser.add_argument(
            "--aspect_ratio_thresh", type=float, default=1.6,
            help="threshold for filtering out boxes of which aspect ratio are above the given value.")
        parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
        return parser


    def perepare_tracker_input(self, bboxes, scores):
        detections = []

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            score = scores[i]
            detections.append([*bbox, score])
        return np.array(detections)

    def track_objects(self, detections):

        if len(detections) == 0:
            return []

        online_targets = self.tracker.update(detections)
        tracked_objects = []
        for track in online_targets:
            track_id = track.track_id
            bbox = track.tlbr  # Top-left to bottom-right format
            tracked_objects.append({
                "track_id": track_id,
                "bbox": bbox.tolist(),
            })
        return tracked_objects