import click
from occupancy import Occupancy
from queue import Queue
import threading
from msghandler import MessageHandler
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


osd_queue = Queue()
msg_queue = Queue()


def inference(source: str, model: str):
    occupancy = Occupancy(source, model,
                          osd_queue=osd_queue,
                          msg_queue=msg_queue)
    occupancy.run()


def event_streaming(bootstrap_server: str, topic: str):
    handler = MessageHandler(bootstrap_server=bootstrap_server,
                             topic=topic)
    while True:
        data = msg_queue.get()
        handler.send_event(**data)


def osd_rtsp_stream(ip="localhost", port=8080):
    """
    Retrieve frames from the queue and stream them using GStreamer.
    
    Parameters:
    ip (str): The IP address of the RTSP server. Default is 'localhost'.
    port (int): The port number for the RTSP server. Default is 8080.
    """
    gst_out = (
        f"appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=ultrafast "
        f"! rtph264pay config-interval=1 pt=96 ! udpsink host={ip} port={port}"
    )

    # GStreamer output setup
    fps = 25  # Adjust based on input stream's FPS
    width, height = 1280, 720  #Adjust based on  input stream resolution

    out = cv2.VideoWriter(
        gst_out,
        cv2.CAP_GSTREAMER,
        0,  # Codec ID (0 for raw pipeline)
        fps,
        (width, height),
    )

    stop_signal = False
    if not out.isOpened():
        logging.error(f"Error: Could not open GStreamer pipeline for output to {ip}:{port}.")
        stop_signal = True
        return

    while not stop_signal:
        # Get a frame from the queue
        frame = osd_queue.get()
        out.write(frame)

    out.release()


@click.command()
@click.option("--source", type=str, help="rtsp link or video path")
@click.option("--model", type=str, help="model artifact")
@click.option("--bootstrap_server", type=str, default="localhost:9092", help="ip:port for kafka broker")
@click.option("--topic", type=str, help="kafka topic where to push event")
@click.option("--ip_rtsp", type=str, default="localhost",
              help="ip for output rtsp stream")
@click.option("--port_rtsp", type=str, default="8080", help="port for output rtsp stream")
def main(source, model, bootstrap_server, topic, ip_rtsp, port_rtsp):
    inference_thread = threading.Thread(target=inference,
                                        kwargs={"model": model,
                                                "source": source})
    # event_streaming_thread = threading.Thread(target=event_streaming,
    #                                          kwargs={"bootstrap_server":
    #                                                  bootstrap_server,
    #                                                  "topic": topic})

    osd_rtsp_stream_thread = threading.Thread(target=osd_rtsp_stream,
                                              kwargs={"port": port_rtsp,
                                                      "ip": ip_rtsp})

    inference_thread.start()
    # event_streaming_thread.start()
    osd_rtsp_stream_thread.start()

    inference_thread.join()
    # event_streaming_thread.join()
    osd_rtsp_stream_thread.join()


if __name__ == "__main__":
    main()