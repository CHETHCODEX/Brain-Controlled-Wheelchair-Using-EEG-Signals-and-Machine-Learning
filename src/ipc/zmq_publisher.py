"""Publish simulation commands on the local machine only."""

import argparse
import time
import uuid

import zmq

from src.ipc.command_schema import NAVIGATION_INTENTS, make_command

ENDPOINT = "tcp://127.0.0.1:5555"


class Publisher:
    def __init__(self, endpoint=ENDPOINT, source="legacy"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.sequence = 0
        self.session_id = uuid.uuid4().hex
        self.source = source
        try:
            self.socket.bind(endpoint)
        except Exception:
            self.close()
            raise

    def send(self, probabilities, **status):
        packet = make_command(probabilities, self.sequence, session_id=self.session_id,
                              source=self.source, **status)
        self.socket.send_json(packet)
        self.sequence += 1
        return packet

    def close(self):
        self.socket.close()
        self.context.term()


def intent_probabilities(intent):
    if intent not in NAVIGATION_INTENTS:
        raise ValueError("Unknown intent")
    return {name: float(name == intent) for name in NAVIGATION_INTENTS}


def scripted_command(intent, duration):
    if not 0 < duration <= 60:
        raise ValueError("Duration must be between zero and 60 seconds")
    publisher = Publisher()
    try:
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            print(publisher.send(intent_probabilities(intent)), flush=True)
            time.sleep(0.25)
    finally:
        publisher.send(intent_probabilities("REST"))
        publisher.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent", choices=NAVIGATION_INTENTS, default="REST")
    parser.add_argument("--duration", type=float, default=5.0)
    arguments = parser.parse_args()
    try:
        scripted_command(arguments.intent, arguments.duration)
    except KeyboardInterrupt:
        pass