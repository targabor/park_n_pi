#!/usr/bin/env python3

import zmq
import time
import pickle
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry


class RobotZMQInterface(Node):
    """
    ZeroMQ interface for a robot in a RL training environment.
    Acts as a Dealer in the ZMQ Dealer-Router pattern.
    """

    def __init__(self):
        """
        Initialize the robot's ZMQ interface.

        Args:
            robot_id: Unique identifier for this robot
            coordinator_address: ZMQ address of the coordinator (router)
        """

        # Initialize ROS node
        super().__init__(f'robot_rl_interface')

        # Parameters
        robot_id = self.declare_parameter('robot_id', 0).value
        self.coordinator_address = self.declare_parameter(
            'coordinator_address', 'tcp://127.0.0.1:5556'
        ).value
        self.get_logger().info(f'Robot ID: {robot_id}, Coordinator: {self.coordinator_address}')
        # Robot identification
        self.robot_id = str(robot_id)
        self.get_logger().info(f'Initializing Robot ZMQ Interface for robot {self.robot_id}')

        # Episode tracking
        self.episode_start_time = 0
        self.episode_max_duration = 60.0  # 1 minutes per episode
        self.episode_id = 0
        self.is_episode_active = False

        # ZeroMQ setup
        self.zmq_context = zmq.Context()
        self.dealer_socket = self.zmq_context.socket(zmq.DEALER)
        self.dealer_socket.setsockopt(zmq.IDENTITY, self.robot_id.encode())
        self.dealer_socket.connect(self.coordinator_address)
        self.is_robot_connected = False
        self.handshake_attempts = 0
        # Create ZMQ poller for non-blocking receives
        self.poller = zmq.Poller()
        self.poller.register(self.dealer_socket, zmq.POLLIN)
        # ROS Topics - Subscribers
        self.lidar_sub = self.create_subscription(
            LaserScan, f'/RaspbotV2_{robot_id}/top_lidar', self.lidar_callback, 10
        )

        self.odom_sub = self.create_subscription(
            Odometry, f'/RaspbotV2_{robot_id}/odom', self.odom_callback, 10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped, f'/RaspbotV2_{robot_id}/goal_pose', self.goal_callback, 10
        )

        # ROS Topics - Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, f'/RaspbotV2_{robot_id}/cmd_vel', 10)

        # State variables
        self.current_lidar = None
        self.current_odom = None
        self.goal_position = None
        self.distance_to_goal = float('inf')

        # Create timer for main control loop
        self.handshake_timer = self.create_timer(2.0, self.check_handshake)
        self.get_logger().info(f'Robot {self.robot_id} ZMQ interface initialized')

    def check_handshake(self):
        """
        Check if the robot has successfully connected to the coordinator.
        If not, attempt to reconnect.
        """
        if self.handshake_attempts >= 10:  # Try 10 times, then give up
            self.get_logger().error("Failed to connect to RL coordinator after multiple attempts")
            return

        self.get_logger().info(
            f"Attempting to connect to RL coordinator (Attempt {self.handshake_attempts + 1}/10)"
        )

        # Send just the HELLO message type - ZMQ will automatically prepend the socket identity
        self.dealer_socket.send_multipart([b'HELLO'])
        socks = dict(self.poller.poll(1000))  # Wait for 1 sec
        self.get_logger().info(f"Poll result: {socks}")

        if self.dealer_socket in socks:
            # Receive the multipart message
            response = self.dealer_socket.recv_multipart()
            if len(response) > 0 and response[0] == b'WELCOME':
                self.get_logger().info(
                    f"Successfully connected to RL coordinator as {self.robot_id}"
                )
                self.handshake_timer.cancel()  # Stop retrying
                self.timer = self.create_timer(0.1, self.control_loop)  # Start control loop
                self.is_robot_connected = True
        else:
            self.handshake_attempts += 1

    def lidar_callback(self, msg):
        """Store the latest lidar scan"""
        self.current_lidar = msg.ranges

    def odom_callback(self, msg):
        """Store the latest odometry and calculate distance to goal"""
        self.current_odom = msg

        if self.goal_position is not None:
            robot_x = msg.pose.pose.position.x
            robot_y = msg.pose.pose.position.y
            goal_x = self.goal_position.pose.position.x
            goal_y = self.goal_position.pose.position.y

            self.distance_to_goal = np.sqrt((robot_x - goal_x) ** 2 + (robot_y - goal_y) ** 2)

    def goal_callback(self, msg):
        """Store the goal position"""
        self.goal_position = msg

    def get_observation(self):
        """
        Construct the observation to send to the RL agent.

        Returns:
            dict: Observation containing lidar, goal position, distance, and odometry
        """
        if self.current_lidar is None or self.current_odom is None or self.goal_position is None:
            return None

        # Normalize the lidar readings
        lidar_data = np.array(self.current_lidar, dtype=np.float32)
        lidar_data = np.clip(lidar_data, 0.0, 8.0)  # Clip to reasonable range

        # Extract robot's position and orientation
        pos_x = self.current_odom.pose.pose.position.x
        pos_y = self.current_odom.pose.pose.position.y

        # Calculate relative goal position
        goal_x = self.goal_position.pose.position.x - pos_x
        goal_y = self.goal_position.pose.position.y - pos_y

        # Get robot's current velocity
        vel_x = self.current_odom.twist.twist.linear.x
        vel_y = self.current_odom.twist.twist.linear.y
        vel_ang = self.current_odom.twist.twist.angular.z

        return {
            'lidar': lidar_data,
            'goal_relative': [goal_x, goal_y],
            'distance_to_goal': self.distance_to_goal,
            'velocity': [vel_x, vel_y, vel_ang],
            'position': [pos_x, pos_y],
            'robot_id': self.robot_id,
            'episode_id': self.episode_id,
            'episode_time': time.time() - self.episode_start_time,
        }

    def execute_action(self, action):
        """
        Execute the received action by publishing to cmd_vel.

        Args:
            action: [linear_x, angular_z] velocities
        """
        cmd = Twist()
        cmd.linear.x = float(action[0])  # Forward/backward velocity
        cmd.angular.z = float(action[1])  # Rotational velocity

        self.cmd_vel_pub.publish(cmd)

    def set_robot_idle(self):
        """Stop the robot by sending zero velocity"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = 0.0

        self.cmd_vel_pub.publish(cmd)

    def send_message(self, message_type, payload):
        """
        Send a message to the coordinator.

        Args:
            message_type: String identifier for the message type
            payload: Data to send (will be pickled)
        """
        self.dealer_socket.send_multipart([message_type.encode(), pickle.dumps(payload)])

    def receive_message(self, timeout=500):
        """
        Attempt to receive a message from the coordinator.

        Args:
            timeout: Time to wait for a message in milliseconds

        Returns:
            tuple or None: (message_type, payload) if a message was received, None otherwise
        """
        socks = dict(self.poller.poll(timeout))
        if self.dealer_socket in socks and socks[self.dealer_socket] == zmq.POLLIN:
            message = self.dealer_socket.recv_multipart()
            message_type = message[0].decode()
            payload = pickle.loads(message[1])
            return message_type, payload
        return None

    def send_observation(self):
        """Send the current observation to the coordinator"""
        observation = self.get_observation()
        if observation is not None:
            self.send_message('OBSERVATION', observation)
            return True
        return False

    def send_episode_done(self, status, reward=0.0):
        """
        Signal that the episode is complete.

        Args:
            status: Reason for episode completion ('SUCCESS', 'COLLISION', 'TIMEOUT')
            reward: Final reward for the episode
        """
        self.is_episode_active = False
        result = {
            'robot_id': self.robot_id,
            'episode_id': self.episode_id,
            'status': status,
            'final_reward': reward,
            'distance_to_goal': self.distance_to_goal,
            'episode_duration': time.time() - self.episode_start_time,
        }

        self.send_message('EPISODE_DONE', result)
        self.get_logger().info(f'Episode {self.episode_id} completed with status: {status}')

    def reset_robot(self):
        """Reset the robot state for a new episode"""
        self.set_robot_idle()
        self.episode_start_time = time.time()
        self.is_episode_active = True
        self.get_logger().info(f'Robot {self.robot_id} reset for episode {self.episode_id}')

    def wait_for_reset(self):
        """Wait until coordinator sends a reset signal"""
        self.get_logger().info(f'Robot {self.robot_id} waiting for reset signal...')

        while rclpy.ok():
            result = self.receive_message(1000)  # Check every second
            if result is not None:
                message_type, payload = result
                if message_type == 'RESET':
                    self.episode_id = payload.get('episode_id', self.episode_id + 1)
                    self.reset_robot()
                    break
            # Spin once to handle ROS callbacks
            rclpy.spin_once(self, timeout_sec=0.1)

    def check_episode_status(self):
        """Check if the current episode should end"""
        current_time = time.time()

        # Check for timeout
        if current_time - self.episode_start_time > self.episode_max_duration:
            self.send_episode_done('TIMEOUT')
            return False

        # Check for goal reached
        if self.distance_to_goal < 0.1:  # 10cm to goal is success
            self.send_episode_done('SUCCESS', reward=1.0)
            return False

        # Check for collision (simple check using minimum lidar reading)
        if self.current_lidar is not None and min(self.current_lidar) < 0.1:  # 10cm to obstacle
            self.send_episode_done('COLLISION', reward=-10.0)
            return False

        return True

    def control_loop(self):
        """Main control loop that handles the RL observation-action cycle"""
        if not self.is_episode_active:
            # If not in an active episode, wait for reset from coordinator
            self.wait_for_reset()
            return

        # Check if episode should continue
        if not self.check_episode_status():
            self.wait_for_reset()
            return

        # Send observation and get action
        if self.send_observation():
            # Wait for action from coordinator
            result = self.receive_message(100)  # Wait up to 100ms for action

            if result is not None:
                message_type, payload = result
                if message_type == 'ACTION':
                    self.execute_action(payload)
                elif message_type == 'RESET':
                    self.episode_id = payload.get('episode_id', self.episode_id + 1)
                    self.reset_robot()


def main(args=None):
    rclpy.init(args=args)
    interface = RobotZMQInterface()

    try:
        rclpy.spin(interface)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        interface.set_robot_idle()
        interface.dealer_socket.close()
        interface.zmq_context.term()
        interface.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
