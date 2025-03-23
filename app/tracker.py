from bytetracker import BYTETracker
import numpy as np


class ObjectTracker:
    def __init__(self, **kwargs):
        self.tracker = BYTETracker(track_thresh=kwargs["track_thresh"], 
                                   match_thresh=kwargs["match_thresh"])

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

        online_targets = self.tracker.update(detections, image_size=image_size)
        tracked_objects = []
        for track in online_targets:
            track_id = track.track_id
            bbox = track.tlbr  # Top-left to bottom-right format
            tracked_objects.append({
                "track_id": track_id,
                "bbox": bbox.tolist(),
            })
        return tracked_objects