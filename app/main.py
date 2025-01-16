import click
from occupancy import Occupancy
from queue import Queue
import threading
from msghandler import MessageHandler
import os
import logging
import ffmpeg

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


def out_rtsp_stream(ip, port, width=640, height=480, fps=30):
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', r=fps)
        .output(f'rtp://{ip}:{port}', vcodec='libx264', pix_fmt='yuv420p', r=fps, preset='ultrafast', tune='zerolatency')
        .run_async(pipe_stdin=True)
    )
    try:
        while True:
            frame = osd_queue.get()
            # Write the frame to FFmpeg's stdin
            process.stdin.write(frame.tobytes())
    except Exception as e:
        print(f"Error during streaming: {e}")
    finally:
        # Close FFmpeg stdin and wait for the process to finish
        process.stdin.close()
        process.wait()


@click.command()
@click.option("--source", type=str, help="rtsp link or video path")
@click.option("--model", type=str, help="model artifact")
@click.option("--bootstrap_server", type=str, default="localhost:9092", help="ip:port for kafka broker")
@click.option("--topic", type=str, help="kafka topic where to push event")
@click.option("--ip_rtsp", type=str, default="localhost",
              help="ip for output rtsp stream")
@click.option("--port_rtsp", type=str, default="8080", help="port for output rtsp stream")
def main(source, model, bootstrap_server, topic, ip_rtsp, port_rtsp):
    source = os.getenv("source", source)
    bootstrap_server = os.getenv("bootstrap_server", bootstrap_server) 
    topic = os.getenv("topic", topic)
    ip_rtsp = os.getenv("ip_rtsp", ip_rtsp)
    port_rtsp = os.getenv("ip_rtsp", port_rtsp)

    inference_thread = threading.Thread(target=inference,
                                        kwargs={"model": model,
                                                "source": source})
    #event_streaming_thread = threading.Thread(target=event_streaming,
    #                                          kwargs={"bootstrap_server":
    #                                                  bootstrap_server,
    #                                                  "topic": topic})
    osd_rtsp_stream_thread = threading.Thread(target=out_rtsp_stream,
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