import gymnasium as gym
from gymnasium import spaces
import rclpy
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from controller import Supervisor
from raspbot_rl_interface.srv import SetRobotPose


LIDAR_MAX_RANGE = 8.0
LIDAR_MIN_RANGE = 0.0
MAP_X_MIN = -5.0
MAP_X_MAX = 5.0
MAP_Y_MIN = -5.0
MAP_Y_MAX = 5.0
MAP_YAW_MIN = -np.pi
MAP_YAW_MAX = np.pi
OMEGA_MIN = -10.0
OMEGA_MAX = 10.0
LINEAR_MIN = -10.0
LINEAR_MAX = 10.0
GOAL_X_MIN = -5.0
GOAL_X_MAX = 5.0
GOAL_Y_MIN = -5.0
GOAL_Y_MAX = 5.0
GOAL_YAW_MIN = -np.pi
GOAL_YAW_MAX = np.pi


class RaspbotEnv(gym.Env):
    def __init__(self, namespace='/RaspbotV2_0'):
        rclpy.init()
        self.namespace = namespace
        self.node = rclpy.create_node(f'{namespace}_env')
        self.timeout_sec = 1.0 / 32.5

        self.observation_space = spaces.Dict(
            {
                "lidar": gym.spaces.Box(
                    low=0.0, high=10.0, shape=(720,), dtype=np.float32
                ),  # LiDAR scan
                # "robot_pose": gym.spaces.Box(
                #     low=np.array([MAP_X_MIN, MAP_Y_MIN, MAP_YAW_MIN, LINEAR_MIN, OMEGA_MIN]),
                #     high=np.array([MAP_X_MAX, MAP_Y_MAX, MAP_YAW_MAX, LINEAR_MAX, OMEGA_MAX]),
                #     dtype=np.float32,
                # ),
                "goal_relative": gym.spaces.Box(
                    low=np.array([MAP_X_MIN, MAP_Y_MIN, -np.pi]),
                    high=np.array([MAP_X_MAX, MAP_Y_MAX, np.pi]),
                    dtype=np.float32,
                ),
            }
        )

        self.action_space = spaces.Box(
            low=-3.75, high=3.75, shape=(2,), dtype=np.float32
        )  # (linear velocity, angular velocity)

        # Publisher to move the robot
        self.cmd_vel_pub = self.node.create_publisher(Twist, f'{namespace}/cmd_vel', 10)
        self.tf_broadcaster = TransformBroadcaster(self.node)

        # Subscribe to get the observations
        self.lidar_sub = self.node.create_subscription(
            LaserScan, f'{namespace}/top_lidar', self.lidar_callback, 10
        )
        self.odom_sub = self.node.create_subscription(
            Odometry, f'{namespace}/odom', self.odom_callback, 10
        )

        # Teleport service to reset the robot's position
        self.teleport_client = self.node.create_client(
            SetRobotPose, '/rl_supervisor/teleport_robot'
        )

        # Initialize the observations
        self.lidar_data = np.zeros(720)
        self.robot_pose = np.zeros(5)
        self.goal_pose = np.zeros(3)
        self.distance_to_goal = np.zeros(3)
        self.prev_distance_to_goal = np.zeros(3)

        self.prev_positions = []  # List to store positions from previous steps
        self.position_history_length = 10  # Number of steps to keep track of
        self.position_threshold = 0.4  # Threshold for position change (e.g., 0.4 meters)
        self.time_limit = 1000  # Maximum number of steps before termination
        self.current_step = 0  # Current step in the episode

    def teleport_robot(self, robot_name, x, y, z, yaw):
        request = SetRobotPose.Request()
        request.robot_name = robot_name
        request.x = x
        request.y = y
        request.z = z
        request.yaw = yaw
        future = self.teleport_client.call_async(request)
        future.add_done_callback(self.handle_teleport_response)

    def handle_teleport_response(self, future):
        try:
            response = future.result()
            self.node.get_logger().info(f"Teleport response: {response.success}")
        except Exception as e:
            self.node.get_logger().error(f"Service call failed: {e}")

    def _publish_goal_tf(self):
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = f"{self.namespace}/base_link"
        t.child_frame_id = f"{self.namespace}/goal"

        # Goal relative to the robot
        t.transform.translation.x = float(self.distance_to_goal[0])
        t.transform.translation.y = float(self.distance_to_goal[1])
        t.transform.translation.z = 0.0

        self.tf_broadcaster.sendTransform(t)

    def quat_to_yaw(self, quat):
        r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        return r.as_euler('xyz')[2]

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)

        # Remove NaNs and clip invalid values
        ranges = np.nan_to_num(
            ranges, nan=LIDAR_MAX_RANGE, posinf=LIDAR_MAX_RANGE, neginf=LIDAR_MIN_RANGE
        )

        # Downsample from 4000 to 720 points
        step = len(ranges) // 720
        self.lidar_data = ranges[::step][:720]

        # Ensure exactly 720 points
        if len(self.lidar_data) < 720:
            self.lidar_data = np.pad(
                self.lidar_data,
                (0, 720 - len(self.lidar_data)),
                'constant',
                constant_values=LIDAR_MAX_RANGE,
            )

    def odom_callback(self, msg):
        self.robot_pose[0] = msg.pose.pose.position.x
        self.robot_pose[1] = msg.pose.pose.position.y
        self.robot_pose[2] = self.quat_to_yaw(msg.pose.pose.orientation)
        self.robot_pose[3] = msg.twist.twist.linear.x
        self.robot_pose[4] = msg.twist.twist.angular.z

    def _get_obs(self):
        # Compute goal distance & angle relative to robot's frame
        dx = self.goal_pose[0] - self.robot_pose[0]
        dy = self.goal_pose[1] - self.robot_pose[1]
        goal_angle = np.arctan2(dy, dx) - self.robot_pose[2]
        # Normalize angle to [-pi, pi]
        goal_angle = (goal_angle + np.pi) % (2 * np.pi) - np.pi

        return {
            "lidar": self.lidar_data,
            "goal_relative": np.array([dx, dy, goal_angle], dtype=np.float32),
        }

    def _get_info(self):
        return {}

    def reset(self, seed=None):
        super().reset(seed=seed)

        fixed_points = [
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

        # Stop the robot
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)

        rclpy.spin_once(self.node, timeout_sec=self.timeout_sec)

        # Ensure ROS messages are processed
        for _ in range(5):
            rclpy.spin_once(self.node, timeout_sec=self.timeout_sec)

        # Reset robot position
        # Pick a random fixed point
        fixed_point = fixed_points[np.random.randint(0, len(fixed_points))]
        random_yaw = np.random.uniform(-np.pi, np.pi)
        self.teleport_robot(
            self.namespace, float(fixed_point[0]), float(fixed_point[1]), 0.0, random_yaw
        )

        # Wait for the robot to teleport
        for _ in range(10):
            rclpy.spin_once(self.node, timeout_sec=self.timeout_sec)

        # Generate a new goal position (make sure it's at least 1m away)
        goal_x = np.random.uniform(-3, 3)
        goal_y = np.random.uniform(-3, 3)
        goal_yaw = np.random.uniform(-np.pi, np.pi)

        # Clip goal to map boundaries
        goal_x = np.clip(self.robot_pose[0] + goal_x, MAP_X_MIN, MAP_X_MAX)
        goal_y = np.clip(self.robot_pose[1] + goal_y, MAP_Y_MIN, MAP_Y_MAX)
        goal_yaw = np.clip(self.robot_pose[2] + goal_yaw, MAP_YAW_MIN, MAP_YAW_MAX)

        self.goal_pose = np.array([goal_x, goal_y, goal_yaw])
        self.distance_to_goal = self.goal_pose - self.robot_pose[:3]
        self.prev_distance_to_goal = self.distance_to_goal

        self.node.get_logger().info(f"New goal position [{self.namespace}]: {self.goal_pose}")
        self.current_step = 0
        obs = self._get_obs()
        info = self._get_info()
        self._publish_goal_tf()
        self.node.get_logger().info(f"{self.namespace} reset")
        return obs, info

    def step(self, action):
        # Publish the action
        cmd_vel = Twist()
        cmd_vel.linear.x = float(action[0])
        cmd_vel.angular.z = float(action[1])
        self.cmd_vel_pub.publish(cmd_vel)
        # Ensure ROS messages are processed
        for _ in range(5):
            rclpy.spin_once(self.node, timeout_sec=self.timeout_sec)

        self.current_step += 1
        info = self._get_info()
        observation = self._get_obs()
        self.prev_distance_to_goal = self.distance_to_goal
        self.distance_to_goal = observation["goal_relative"]
        # Use absolute reduction in distance
        distance_before = np.linalg.norm(self.prev_distance_to_goal[:2])
        distance_after = np.linalg.norm(self.distance_to_goal[:2])
        change_in_distance = distance_before - distance_after
        # Check termination conditions
        collision = np.any(self.lidar_data < 0.12)
        reached_goal = np.linalg.norm(self.distance_to_goal) <= 0.15
        done = reached_goal or collision or self.current_step >= self.time_limit
        truncated = False
        # Improved Reward Function
        if reached_goal:
            reward = 250.0  # Significant reward for reaching goal
        elif collision:
            reward = -100.0  # More significant penalty for collision
        else:
            # Reward based on progress towards goal
            reward = 10.0 * change_in_distance

            # Additional penalties for non-progressive behavior
            if change_in_distance < 0:  # Moving away from goal
                reward -= 5.0

            # Encourage movement
            if np.abs(cmd_vel.linear.x) < 0.1 and np.abs(cmd_vel.angular.z) < 0.1:
                reward -= 10.0

            # Penalize large angular velocities
            reward -= 0.1 * np.abs(action[1])
        self._publish_goal_tf()
        return observation, float(reward), done, truncated, info
