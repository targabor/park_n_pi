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
MAP_X_MIN = -10.0
MAP_X_MAX = 10.0
MAP_Y_MIN = -10.0
MAP_Y_MAX = 10.0 
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
                    low=0.0, high=10.0, shape=(180,), dtype=np.float32
                ),
                # "robot_pose": gym.spaces.Box(
                #     low=np.array([MAP_X_MIN, MAP_Y_MIN, MAP_YAW_MIN]),
                #     high=np.array([MAP_X_MAX, MAP_Y_MAX, MAP_YAW_MAX]),
                #     dtype=np.float32,
                # ),
                # "goal_pose": gym.spaces.Box(
                #     low=np.array([GOAL_X_MIN, GOAL_Y_MIN, GOAL_YAW_MIN]),
                #     high=np.array([GOAL_X_MAX, GOAL_Y_MAX, GOAL_YAW_MAX]),
                #     dtype=np.float32,
                # ),
                "goal_relative": gym.spaces.Box(
                    low=np.array([-20.0, -20.0, -np.pi]),
                    high=np.array([20.0, 20.0, np.pi]),
                    dtype=np.float32,
                ),
                "goal_distance": gym.spaces.Box(
                    low=np.array([0.0]),
                    high=np.array([50.0]),
                    dtype=np.float32,
                ),
            }
        )


        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -4.0]),  # forward movement
            high=np.array([6.0, 6.0]),   # forward faster
            dtype=np.float32
        )


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

        # Set robot pose after teleportation
        self.robot_pose_pub = self.node.create_publisher(
            Odometry, f'{namespace}/pose_update', 10
        )

        # Initialize the observations
        self.lidar_data = np.zeros(180)
        self.robot_pose = np.zeros(3)
        self.goal_pose = np.zeros(3)


        self.position_history_length = 10  # Number of steps to keep track of
        self.position_threshold = 0.4  # Threshold for position change (e.g., 0.4 meters)
        self.time_limit = 2000  # Maximum number of steps before termination
        self.current_step = 0  # Current step in the episode

        self.new_x = 0.0
        self.new_y = 0.0
        self.new_yaw = 0.0

        self.phase = 1  # Set to True to enable tuning mode
        self.total_reward = 0.0  # Total reward accumulated in the episode
        self.success_count = 0  # Counter for successful goal reaches
        self.stuck_count = 0  # Counter for stuck episodes
    def teleport_robot(self, robot_name, x, y, z, yaw):
        self.new_x = x
        self.new_y = y
        self.new_yaw = yaw

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
            if response is None:
                self.node.get_logger().error("Service call failed")
                return
            if response.success:
                #self.node.get_logger().info("Teleport successful")

                r = R.from_euler('z', self.new_yaw, degrees=False)
                quat = r.as_quat()  # Returns (x, y, z, w)

                odometry = Odometry()
                odometry.header.stamp = self.node.get_clock().now().to_msg()
                odometry.header.frame_id = f"{self.namespace}/base_link"
                odometry.child_frame_id = f"{self.namespace}/base_link"
                odometry.pose.pose.position.x = self.new_x
                odometry.pose.pose.position.y = self.new_y
                odometry.pose.pose.orientation.x = quat[0]
                odometry.pose.pose.orientation.y = quat[1]
                odometry.pose.pose.orientation.z = quat[2]
                odometry.pose.pose.orientation.w = quat[3]


                self.robot_pose_pub.publish(odometry)
                #self.node.get_logger().info("Odometry published")
        except Exception as e:
            self.node.get_logger().error(f"Service call failed: {e}")

    def _publish_goal_tf(self):
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = f"{self.namespace}/odom"
        t.child_frame_id = f"{self.namespace}/goal"

        # Absolute goal position in odom/map frame
        t.transform.translation.x = float(self.goal_pose[0])
        t.transform.translation.y = float(self.goal_pose[1])
        t.transform.translation.z = 0.0

        # Identity rotation (no orientation)
        t.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t)
        

    def quat_to_yaw(self, quat):
        r = R.from_quat([quat.x, quat.y, quat.z, quat.w])
        return r.as_euler('xyz')[2]
    
    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi


    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)

        # Clean invalid values
        ranges = np.nan_to_num(
            ranges, nan=LIDAR_MAX_RANGE, posinf=LIDAR_MAX_RANGE, neginf=LIDAR_MIN_RANGE
        )

        total_points = len(ranges)

        # Extract forward-facing 180° (half of the scan)
        # Assuming angle 0 is forward, we take 90° to the left and 90° to the right
        start = total_points // 4     # 90° left
        end = 3 * total_points // 4   # 90° right
        forward_ranges = ranges[start:end]

        # Downsample to 180 points
        step = max(1, len(forward_ranges) // 180)
        self.lidar_data = forward_ranges[::step][:180]

        # Ensure exactly 180 elements
        if len(self.lidar_data) < 180:
            self.lidar_data = np.pad(
                self.lidar_data,
                (0, 180 - len(self.lidar_data)),
                'constant',
                constant_values=LIDAR_MAX_RANGE,
            )
            
        assert self.lidar_data.shape == (180,), f"Lidar shape mismatch: {self.lidar_data.shape}"


        
    def odom_callback(self, msg):
        self.robot_pose[0] = msg.pose.pose.position.x
        self.robot_pose[1] = msg.pose.pose.position.y
        self.robot_pose[2] = self.quat_to_yaw(msg.pose.pose.orientation)

    def _get_obs(self):
        # Vector to goal in world frame
        dx = self.goal_pose[0] - self.robot_pose[0]
        dy = self.goal_pose[1] - self.robot_pose[1]

        # Robot orientation
        yaw = self.robot_pose[2]

        # Rotate (dx, dy) into robot's local frame
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        local_dx = dx * cos_yaw - dy * sin_yaw
        local_dy = dx * sin_yaw + dy * cos_yaw

        # Relative angle to goal (in robot's frame)
        goal_angle = np.arctan2(local_dy, local_dx)
        goal_angle = (goal_angle + np.pi) % (2 * np.pi) - np.pi

        # Euclidean distance
        goal_distance = np.hypot(local_dx, local_dy)
        
        # print(f"[OBS DEBUG] robot: {self.robot_pose} | goal: {self.goal_pose}")
        # print(f"[OBS DEBUG] dx={dx:.2f}, dy={dy:.2f}, yaw={yaw:.2f}")
        # print(f"[OBS DEBUG] local_dx={local_dx:.2f}, local_dy={local_dy:.2f}, angle={goal_angle:.2f}, distance={goal_distance:.2f}")

        return {
            "lidar": self.lidar_data,
            # "robot_pose": self.robot_pose,
            # "goal_pose": self.goal_pose,
            "goal_relative": np.array([local_dx, local_dy, goal_angle], dtype=np.float32),
            "goal_distance": np.array([goal_distance], dtype=np.float32),
        }

    def _get_info(self):
        return {}

    def reset(self, seed=None):
        super().reset(seed=seed)

        fixed_points = [
            (-7.5, -7.5), (-7.5, -2.5), (-7.5, 2.5), (-7.5, 7.5),
            (-2.5, -7.5), (-2.5, -2.5), (-2.5, 2.5), (-2.5, 7.5),
            (2.5, -7.5),  (2.5, -2.5),  (2.5, 2.5),  (2.5, 7.5),
            (7.5, -7.5),  (7.5, -2.5),  (7.5, 2.5),  (7.5, 7.5),
        ]


        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)
        for _ in range(5):
            rclpy.spin_once(self.node, timeout_sec=self.timeout_sec)

        fixed_point = fixed_points[np.random.randint(0, len(fixed_points))]
        random_yaw = np.random.uniform(-np.pi, np.pi)
        self.teleport_robot(self.namespace, fixed_point[0], fixed_point[1], 0.0, random_yaw)
        for _ in range(10):
            rclpy.spin_once(self.node, timeout_sec=self.timeout_sec)

        if not hasattr(self, 'success_count'):
            self.success_count = 0

        if self.success_count < 20:
            max_dist, angle_range = 2.0, np.pi / 3  # ±60° front sector
        elif self.success_count < 40:
            max_dist, angle_range = 3.0, np.pi / 2  # ±90° front sector
        else:
            max_dist, angle_range = 4.0, np.pi      # 360° around robot

        # Sample distance based on phase
        if self.success_count < 40:
            goal_dist = np.random.uniform(1.0, max_dist)  # always forward
        else:
            goal_dist = np.random.uniform(-max_dist, max_dist)  # full circle
        goal_angle = np.random.uniform(-angle_range, angle_range)

        goal_x = fixed_point[0] + goal_dist * np.cos(random_yaw + goal_angle)
        goal_y = fixed_point[1] + goal_dist * np.sin(random_yaw + goal_angle)
        goal_yaw = np.random.uniform(-np.pi, np.pi)

        goal_x = np.clip(goal_x, -9.5, 9.5)
        goal_y = np.clip(goal_y, -9.5, 9.5)

        self.goal_pose = np.array([goal_x, goal_y, goal_yaw])

        self.node.get_logger().info(f"New goal position [{self.namespace}]: {self.goal_pose}")

        self.current_step = 0
        self.total_reward = 0.0

        self.start_position = np.copy(self.robot_pose)

        obs = self._get_obs()
        info = self._get_info()
        
        self.prev_obs = obs.copy()
        self._publish_goal_tf()
        self.node.get_logger().info(f"{self.namespace} reset")

        return obs, info


    def step(self, action):
        cmd_vel = Twist()
        cmd_vel.linear.x = float(action[0])
        cmd_vel.angular.z = float(action[1])
        self.cmd_vel_pub.publish(cmd_vel)

        for _ in range(5):
            rclpy.spin_once(self.node, timeout_sec=self.timeout_sec)

        self.current_step += 1

        info = self._get_info()
        observation = self._get_obs()

        progress_scale = 80.0
        collision_penalty_value = 150.0
        goal_reward_value = 450.0
        goal_tolerance = 0.2
        collision_distance = 0.19

        lidar = observation['lidar']
        distance = observation['goal_distance']
        goal_relative = observation['goal_relative']
        prev_relative = self.prev_obs['goal_relative']
        # Progress reward
        curr_dist = distance[0]
        prev_dist = self.prev_obs['goal_distance'][0]
        progress_reward = 0.0 if self.current_step < 5 else progress_scale * (prev_dist - curr_dist)
        self.prev_obs = observation.copy()

        # Collision
        min_lidar = np.min(lidar)
        is_collision = min_lidar < collision_distance and self.current_step > 5
        collision_penalty = -collision_penalty_value if is_collision else 0.0

        # --- Yaw alignment reward (reward turning *toward* the goal) ---
        yaw_reward = 0.0
        if self.current_step >= 5 and min_lidar > 0.3:  # don't reward yaw near obstacles
            yaw_error_curr = abs(goal_relative[2])
            yaw_error_prev = abs(prev_relative[2])
            yaw_alignment_scale = 15.0
            yaw_reward = yaw_alignment_scale * (yaw_error_prev - yaw_error_curr)

        
        # Yaw alignment reward
        # yaw_error_curr = abs(goal_relative[2])
        # yaw_error_prev = abs(prev_relative[2])
        # yaw_alignment_scale = 10.0
        # yaw_reward = 0.0 if self.current_step < 5 else yaw_alignment_scale * (yaw_error_prev - yaw_error_curr)


        # Goal reached
        at_position = curr_dist < goal_tolerance
        reached_goal = at_position  # skip yaw for now
        goal_reward = goal_reward_value if reached_goal else 0.0
        
        # Small penalty for each step
        step_penalty = -0.05

        # Final reward
        reward = progress_reward + collision_penalty + goal_reward + yaw_reward + step_penalty


        truncated = self.current_step >= self.time_limit
        done = reached_goal or is_collision or truncated

        self.prev_obs = observation.copy()
        self.total_reward += reward
        if done:
            info["episode"] = {
                "r": self.total_reward,
                "l": self.current_step,
            }
            info["is_success"] = float(reached_goal)
            info["collided"] = int(is_collision)


        self._publish_goal_tf()
        return observation, float(reward), done, truncated, info
