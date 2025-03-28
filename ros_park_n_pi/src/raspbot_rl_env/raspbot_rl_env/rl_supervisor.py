#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
from rosgraph_msgs.msg import Clock
from raspbot_rl_interface.srv import SetRobotPose
import random


class MovingObject:
    def __init__(
        self,
        supervisor,
        def_name,
        cycle_duration=5.0,
        max_distance=2.0,
        direction=[1.0, 0.0, 0.0],
        delay=0.0,
        movement_type="sine",
    ):
        """
        Initialize a moving object with custom parameters

        Parameters:
        - supervisor: The Supervisor instance
        - def_name: DEF name of the object in the world
        - cycle_duration: Time in seconds for one direction (default: 5.0)
        - max_distance: Maximum distance to move from initial position (default: 2.0)
        - direction: Direction vector [x, y, z] (default: [1.0, 0.0, 0.0])
        - delay: Delay in seconds before starting movement (default: 0.0)
        - movement_type: Type of movement function ("sine", "linear", "bounce")
        """
        self.node = supervisor.getFromDef(def_name)
        if self.node is None:
            print(f"Could not find object: {def_name}")
            self.valid = False
            return

        self.valid = True
        self.translation_field = self.node.getField("translation")
        self.initial_position = self.translation_field.getSFVec3f()
        self.cycle_duration = cycle_duration
        self.max_distance = max_distance
        self.direction = direction
        self.delay = delay
        self.movement_type = movement_type
        self.elapsed_time = -delay  # Negative time for delay

    def update(self, dt):
        """Update object position based on elapsed time"""
        if not self.valid:
            return

        # Update elapsed time
        self.elapsed_time += dt

        # Don't move during delay period
        if self.elapsed_time < 0:
            return

        # Calculate position in cycle (0 to 1 to 0)
        cycle_position = (self.elapsed_time % (self.cycle_duration * 2)) / self.cycle_duration
        if cycle_position > 1.0:
            cycle_position = 2.0 - cycle_position  # Return phase

        # Apply different movement patterns
        if self.movement_type == "sine":
            factor = math.sin(cycle_position * math.pi / 2)
        elif self.movement_type == "linear":
            factor = cycle_position
        elif self.movement_type == "bounce":
            factor = 4 * cycle_position * (1 - cycle_position)  # Quadratic ease-in-out
        else:
            factor = cycle_position  # Default to linear

        # Calculate new position
        new_position = [
            self.initial_position[0] + self.direction[0] * self.max_distance * factor,
            self.initial_position[1] + self.direction[1] * self.max_distance * factor,
            self.initial_position[2] + self.direction[2] * self.max_distance * factor,
        ]

        # Update object position
        self.translation_field.setSFVec3f(new_position)


class WebotsSupervisorDriver:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot
        timestep = int(self.__robot.getBasicTimeStep())
        rclpy.init(args=None)

        self.__node = rclpy.create_node(f"rl_supervisor", namespace=f"rl_supervisor")

        self.timestep = int(self.__robot.getBasicTimeStep())  # Use existing robot
        self.dt = self.timestep / 1000.0  # Convert milliseconds to seconds
        self.__node.get_logger().info(
            f'Webots timestep: {self.timestep}ms ({1000/self.timestep:.2f}Hz)'
        )

        self.clock_publisher_ = self.__node.create_publisher(Clock, "/clock", 10)
        self.__node.create_timer(self.dt, self.step_simulation)
        self.__node.get_logger().info("Supervisor driver initialized")
        # Import the required service interfaces

        # Initialize the service server
        self.teleport_service = self.__node.create_service(
            SetRobotPose, 'teleport_robot', self.handle_teleport_request
        )
        self.__node.get_logger().info("Teleport service creted")

        self.robot_nodes = {}

        self.last_used_coords = []

        self.fixed_points = [
            (-4.2, 4.44),
            (-4.39, 3.13),
            (-4.32, -4.3),
            (-4.3, -4.0),
            (-4.3, -3.8),
            (-4.3, -3.0),
            (-2.9, -2.8),
            (-2.52, -1.2),
            (-2.26, 0.02),
            (-1.33, 1.39),
            (-0.509, 3.26),
            (1.08, 3.24),
            (0.946, 1.94),
            (2.6, 2.01),
            (3.22, 3.06),
            (4.08, 1.25),
            (4.4, -1.03),
            (4.19, -4.02),
            (2.99, -3.69),
            (1.61, -4.09),
            (-0.117, -3.66),
            (-0.0491, -2.5),
            (1.43, -1.22),
        ]

    def handle_teleport_request(self, request, response):
        """Handle teleport request."""
        # Debugging types
        if (request.x, request.y) in self.last_used_coords:
            # Pick another random point, which wasnt used before
            random_point = random.choice(self.fixed_points)
            while random_point in self.last_used_coords:
                random_point = random.choice(self.fixed_points)
            request.x, request.y = random_point

        self.last_used_coords.append((request.x, request.y))
        # Crop last used coords to the last 10
        if len(self.last_used_coords) > 10:
            self.last_used_coords.pop(0)

        print(f"request.x type: {type(request.x)}")
        print(f"request.y type: {type(request.y)}")
        print(f"request.z type: {type(request.z)}")
        print(f"request.yaw type: {type(request.yaw)}")
        robot = self.__robot.getFromDef(request.robot_name)
        if robot is None:
            response.success = False
            response.message = f"Robot '{request.robot_name}' not found"
            self.__node.get_logger().error(response.message)
            return response

        # Set robot position
        translation_field = robot.getField("translation")
        translation_field.setSFVec3f(
            [request.x, request.y, 0.1]
        )  # Add some z to avoid collision with the ground

        # Set robot orientation (Yaw to quaternion)
        rotation_field = robot.getField("rotation")

        # Convert yaw (request.yaw) to quaternion
        yaw = request.yaw  # Assuming yaw is in radians
        w = math.cos(yaw / 2)
        x = 0
        y = 0
        z = math.sin(yaw / 2)

        # Set the new rotation using quaternion [x, y, z, w]
        rotation_field.setSFRotation([x, y, z, w])
        # Return a success response
        response.success = True
        response.message = f"Successfully teleported robot '{request.robot_name}' to position [{request.x}, {request.y}, {request.z}] with yaw {request.yaw} radians."
        self.__node.get_logger().info(response.message)
        return response

    def step_simulation(self):
        """Advance Webots simulation and publish time."""
        self.__robot.step(self.timestep)

        clock_msg = Clock()
        clock_msg.clock.sec = int(self.__robot.getTime())
        clock_msg.clock.nanosec = int((self.__robot.getTime() % 1) * 1e9)
        self.clock_publisher_.publish(clock_msg)

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)
