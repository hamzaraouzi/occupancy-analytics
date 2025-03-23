import numpy as np
from yolox.tracker.byte_tracker import BYTETracker
from argparse import Namespace
import torch
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

    def track_objects(self, bboxs, scores, image_size):

        if bboxs.shape[0] == 0:
            return []

        detections = self.perepare_tracker_input(
            bboxes=bboxs,
            scores=scores
        )
        online_targets = self.tracker.update(torch.from_numpy(detections),
                                             img_size=image_size,
                                             img_info=list(image_size))
        tracked_objects = []
        for track in online_targets:
            track_id = track.track_id
            bbox = track.tlbr  # Top-left to bottom-right format
            tracked_objects.append({
                "track_id": track_id,
                "bbox": bbox.tolist(),
            })
        return tracked_objects