import cv2


def calculate_center(bbox):
    center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    return center


def has_crossed_line(prev_center, current_center, line):
    x1, y1 = prev_center
    x2, y2 = current_center
    (lx1, ly1), (lx2, ly2) = line

    prev_sign = (lx2 - lx1) * (y1 - ly1) - (ly2 - ly1) * (x1 - lx1)
    curr_sign = (lx2 - lx1) * (y2 - ly1) - (ly2 - ly1) * (x2 - lx1)

    return prev_sign * curr_sign < 0


def prepare_osd_frames(frame, bbox, center, line):
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.circle(frame, center, 5, (0, 255, 255), -1)
    cv2.line(frame, line[0], line[1], (0, 0, 255), 2)
    return frame


def update_obj_history(object_histories, object_id, center):
    if object_id not in object_histories:
        object_histories[object_id] = []
    object_histories[object_id].append(center)
    return object_histories