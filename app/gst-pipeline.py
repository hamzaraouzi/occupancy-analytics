import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib
import cv2
import numpy as np

Gst.init(None)

# Create main pipeline
pipeline = Gst.Pipeline()

# Input Source
src = Gst.ElementFactory.make("filesrc", "file-source")
src.set_property("location", "/Users/admin/Desktop/personal-repos/occupancy-analytics/videos/person.mp4")
# Input Queue
queue_in = Gst.ElementFactory.make("queue", "input_queue")

# Detection Element (custom logic via appsink/appsrc)
detect_sink = Gst.ElementFactory.make("appsink", "detect_sink")
detect_sink.set_property("emit-signals", True)
detect_sink.set_property("sync", False)

detect_src = Gst.ElementFactory.make("appsrc", "detect_src")
detect_src.set_property("format", Gst.Format.TIME)
detect_src.set_property("is-live", True)

# Detection Queue
queue_detect = Gst.ElementFactory.make("queue", "detect_queue")

# Tracking Element (custom logic via appsink/appsrc)
track_sink = Gst.ElementFactory.make("appsink", "track_sink")
track_sink.set_property("emit-signals", True)
track_sink.set_property("sync", False)

track_src = Gst.ElementFactory.make("appsrc", "track_src")
track_src.set_property("format", Gst.Format.TIME)
track_src.set_property("is-live", True)

# Output Queue
queue_out = Gst.ElementFactory.make("queue", "output_queue")

# Output Sink
sink = Gst.ElementFactory.make("autovideosink", "display")

# Build pipeline
pipeline.add(src)
pipeline.add(queue_in)
pipeline.add(detect_sink)
pipeline.add(detect_src)
pipeline.add(queue_detect)
pipeline.add(track_sink)
pipeline.add(track_src)
pipeline.add(queue_out)
pipeline.add(sink)

# Link elements
src.link(queue_in)
queue_in.link(detect_sink)

detect_src.link(queue_detect)
queue_detect.link(track_sink)

track_src.link(queue_out)
queue_out.link(sink)

# Buffer processing functions
def gst_to_np(sample):
    buf = sample.get_buffer()
    caps = sample.get_caps()
    structure = caps.get_structure(0)
    width, height = structure.get_value("width"), structure.get_value("height")
    img  =  np.ndarray(
        (height, width, 3),
        buffer=buf.extract_dup(0, buf.get_size()),
        dtype=np.uint8
    )
    print(img.shape, "++++++++")
    return img

def np_to_gst(arr):
    height, width, _ = arr.shape
    caps = Gst.Caps.from_string(f"video/x-raw,format=RGB,width={width},height={height}")
    buf = Gst.Buffer.new_wrapped(arr.tobytes())
    return Gst.Sample.new(buf, caps, None, None)

# Detection logic
def detection_callback(sink):
    sample = sink.emit("pull-sample")
    frame = gst_to_np(sample)

    # Your detection code here (example boxes)
    boxes = [(100, 100, 50, 50), (200, 200, 60, 40)]  # Replace with real detection

    # Add boxes to metadata
    sample = np_to_gst(frame)
    sample.get_buffer().set_meta("boxes", boxes)
    detect_src.emit("push-sample", sample)
    return Gst.FlowReturn.OK

# Tracking logic
def tracking_callback(sink):
    sample = sink.emit("pull-sample")
    frame = gst_to_np(sample)
    boxes = sample.get_buffer().get_meta("boxes")

    # Your tracking code here
    tracked_objects = [(1, (100, 100, 50, 50)), (2, (200, 200, 60, 40))]  # Replace with real tracking

    # Draw bounding boxes
    for obj_id, (x, y, w, h) in tracked_objects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
    
    # Push to output
    sample = np_to_gst(frame)
    track_src.emit("push-sample", sample)
    return Gst.FlowReturn.OK

# Connect callbacks
detect_sink.connect("new-sample", detection_callback)
track_sink.connect("new-sample", tracking_callback)

# Start pipeline
pipeline.set_state(Gst.State.PLAYING)

# Main loop
loop = GLib.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    pass

# Cleanup
pipeline.set_state(Gst.State.NULL)