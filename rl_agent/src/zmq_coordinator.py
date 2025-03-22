import zmq
import pickle
import logging
from typing import Dict


class RLServer:
    def __init__(self, log_level=logging.DEBUG):
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        self.robots = {}  # Dictionary to track multiple robots

        # Set up ZeroMQ router socket
        context = zmq.Context()
        self.router_socket = context.socket(zmq.ROUTER)
        self.router_socket.bind("tcp://127.0.0.1:5556")
        self.logger.info("RLServer initialized and socket bound to tcp://*:5556")

    def register_robot(self, robot_id: str):
        """Register a new robot in the RL system."""
        if robot_id not in self.robots:
            self.robots[robot_id] = {'episode_done': False}
            self.logger.info(f"Registered robot {robot_id}")

    def receive_message(self):
        """Receive a message from a robot and process it."""
        try:
            message = self.router_socket.recv_multipart()
            self.logger.debug(f"Received message: {message}")
            if len(message) < 2:
                self.logger.warning(f"Received malformed message: {message}")
                return

            robot_id = message[0].decode()  # First frame is always the identity
            msg_type = message[1].decode()  # Second frame is the actual message type
            data = (
                message[2] if len(message) > 2 else b''
            )  # Third frame (if present) is the payload

            self.logger.debug(f"Received message from {robot_id}: {msg_type}")
            self.logger.debug(f"Data: {data}")

            if msg_type == 'HELLO':
                self.register_robot(robot_id)
                self.logger.info(f"Registered robot {robot_id}")

                # Reply using the same identity
                self.router_socket.send_multipart([robot_id.encode(), b'WELCOME'])
        except zmq.Again:
            self.logger.debug("No messages received")

    def run(self):
        """Main loop to receive messages and process them."""
        self.logger.info("Starting main server loop")
        while True:
            self.receive_message()


if __name__ == '__main__':
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    server = RLServer()
    server.run()
