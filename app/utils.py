import cv2
import numpy as np

def calculate_center(bbox):
    center = (int((bbox[0] + bbox[2]) // 2), int((bbox[1] + bbox[3]) // 2))
    return center


def has_crossed_line(prev_center, current_center, line):
    if prev_center is None:
        return False, ""

    x1, y1 = prev_center[-1]
    x2, y2 = current_center
    (lx1, ly1), (lx2, ly2) = line

    prev_sign = (lx2 - lx1) * (y1 - ly1) - (ly2 - ly1) * (x1 - lx1)
    curr_sign = (lx2 - lx1) * (y2 - ly1) - (ly2 - ly1) * (x2 - lx1)

    if prev_sign * curr_sign < 0:  # Opposite signs indicate crossing
        # Determine direction based on crossing vector
        direction = (lx2 - lx1) * (y2 - y1) - (ly2 - ly1) * (x2 - x1)
        if direction > 0:
            return True, "entry"
        else:
            return True, "exit"

    return False, ""


def prepare_osd_frames(frame, bbox, center, line, obj_id):
    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
    cv2.putText(frame, f'#{obj_id}', (int(bbox[0]), int(bbox[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(frame, center, 5, (0, 255, 255), -1)
    cv2.line(frame, line[0], line[1], (0, 0, 255), 2)
    return frame


def draw_tracking_bbox(frame, bbox, obj_id):
    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
    cv2.putText(frame, f'#{obj_id}', (int(bbox[0]), int(bbox[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 2, cv2.LINE_AA)
    return frame


def update_obj_history(object_histories, object_id, center):
    if object_id not in object_histories:
        object_histories[object_id] = [center]
    object_histories[object_id].append(center)
    return object_histories


def write_output_video(frames, output_path, fps=30, codec='mp4v'):
    """
    Write a list of frames to a video file.

    Parameters:
        frames (list): List of frames (numpy arrays) to write to video.
        output_path (str): Path to the output video file.
        fps (int): Frames per second for the output video.
        codec (str): Codec to use for video writing (e.g., 'mp4v', 'XVID').
    """
    if not frames:
        print("No frames provided.")
        return

    # Get the frame dimensions
    height, width, _ = frames[0].shape

    # Define the codec and create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        # Ensure all frames are of the same size
        if frame.shape[:2] != (height, width):
            print("Error: Frame dimensions do not match the initial frame.")
            out.release()
            return

        out.write(frame)

    out.release()
    print(f"Video saved at {output_path}")