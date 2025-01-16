import click
from occupancy import Occupancy
from queue import Queue
import threading
from msghandler import MessageHandler
import os
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


def out_rtsp_stream():
    while True:
        frame = osd_queue.get()
        print("render out frame")


@click.command()
@click.option("--source", type=str, help="rtsp link or video path")
@click.option("--model", type=str, help="model artifact")
@click.option("--bootstrap_server", type=str, default="localhost:9092", help="ip:port for kafka broker")
@click.option("--topic", type=str, help="kafka topic where to push event")
def main(source, model, bootstrap_server, topic):
    source = os.getenv("source", source)
    bootstrap_server = os.getenv("bootstrap_server", bootstrap_server) 
    topic = os.getenv("topic", topic)

    inference_thread = threading.Thread(target=inference,
                                        kwargs={"model": model,
                                                "source": source})
    # event_streaming_thread = threading.Thread(target=event_streaming,
    #                                          kwargs={"bootstrap_server":
    #                                                  bootstrap_server,
    #                                                  "topic": topic})

    inference_thread.start()
    # event_streaming_thread.start()

    inference_thread.join()
    # event_streaming_thread.join()


if __name__ == "__main__":
    main()