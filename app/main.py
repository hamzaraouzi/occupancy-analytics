import click
from line_crossing import LineCrossing
from queue import Queue
import threading
from msghandler import MessageHandler
import os
import logging
from typing import List
from object_tracker import ObjectTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

msg_queue = list()

def event_streaming(bootstrap_server: str, topic: str):
    handler = MessageHandler(bootstrap_server=bootstrap_server,
                             topic=topic)
    while True:
        if len(msg_queue) > 0:
            data = msg_queue[0]

            sucess = handler.send_event(**data)
            if sucess:
                msg_queue.pop[0]
            else:
                continue


@click.command()
@click.option("--source", type=str, help="rtsp link or video path")
@click.option("--model", type=str, help="model artifact")
@click.option("--bootstrap_server", type=str, default="localhost:9092", help="ip:port for kafka broker")
@click.option("--topic", type=str, help="kafka topic where to push event")
def main(source, model, bootstrap_server, topic):
    source = os.getenv("source", source)
    bootstrap_server = os.getenv("bootstrap_server", bootstrap_server) 
    topic = os.getenv("topic", topic)
    tracker = ObjectTracker(track_thresh=0.6, match_thresh=0.9,
                            track_buffer=60, mot20=False)
    line = [(750, 200), (950, 1250)]

    line_crossing = LineCrossing(source=source, model=model, tracker=tracker,
                                 line=line, msg_queue=msg_queue)

    #event_streaming_thread = threading.Thread(target=event_streaming,
    #                                          kwargs={"bootstrap_server":
    #                                                  bootstrap_server,
    #                                                 "topic": topic})

    line_crossing.start()
    #event_streaming_thread.start()

    line_crossing.join()
    #event_streaming_thread.join()


if __name__ == "__main__":
    main()