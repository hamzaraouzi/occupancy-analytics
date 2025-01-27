from kafka import KafkaProducer
import logging
import json
import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageHandler:
    def __init__(self, bootstrap_server, topic):
        self.bootstrap_server = bootstrap_server
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_server,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all'
            )

    def send_event(self, direction, cls):
        time_stamp = datetime.datetime.now()
        time_stamp = time_stamp.strftime("%Y-%m-%dT%H:%M:%S")
        data = {"time_stamp": str(time_stamp),
                "direction": direction,
                "obj_class": cls}

        def on_success(record_metadata):
            nonlocal success
            success = True

        def on_error(ex):
            nonlocal success
            success = False
            logger.error("Failed to send message")

        success = None
        future = self.producer.send(self.topic, value=data)
        future.add_callback(on_success).add_errback(on_error)

        return success
