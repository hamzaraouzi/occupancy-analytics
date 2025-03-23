import numpy as np
from yolox.tracker.byte_tracker import BYTETracker
from argparse import Namespace

class ObjectTracker:
    def __init__(self, **kwargs):
        args = Namespace(**kwargs)
        self.tracker = BYTETracker(args=args)

    def perepare_tracker_input(self, bboxes, scores):
        detections = []

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            score = scores[i]
            detections.append([*bbox, score])
        return np.array(detections)

    def track_objects(self, detections, image_size):

        if len(detections) == 0:
            return []

        online_targets = self.tracker.update(detections, img_size=image_size, 
                                             img_info=None)
        tracked_objects = []
        for track in online_targets:
            track_id = track.track_id
            bbox = track.tlbr  # Top-left to bottom-right format
            tracked_objects.append({
                "track_id": track_id,
                "bbox": bbox.tolist(),
            })
        return tracked_objects