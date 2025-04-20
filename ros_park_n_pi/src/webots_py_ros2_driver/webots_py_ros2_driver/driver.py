import rclpy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3, Twist, TransformStamped
from nav_msgs.msg import Odometry
import numpy as np
import math
import tf2_ros
from scipy.spatial.transform import Rotation as R


class Raspbotv2RobotDriver:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot
        self._robot_id = webots_node.robot.name.split("_")[-1]
        timestep = int(self.__robot.getBasicTimeStep())

        self.acceleromter_ = self.__robot.getDevice("accelerometer")
        self.acceleromter_.enable(timestep)

        self.gyroscope_ = self.__robot.getDevice("gyroscope")
        self.gyroscope_.enable(timestep)

        self.magnetometer_ = self.__robot.getDevice("magnetometer")
        self.magnetometer_.enable(timestep)

        self.front_left_motor_ = self.__robot.getDevice("l1_Joint")
        self.front_right_motor_ = self.__robot.getDevice("r1_Joint")
        self.rear_left_motor_ = self.__robot.getDevice("l2_Joint")
        self.rear_right_motor_ = self.__robot.getDevice("r2_Joint")

        self.front_left_motor_.setPosition(float("inf"))
        self.front_right_motor_.setPosition(float("inf"))
        self.rear_left_motor_.setPosition(float("inf"))
        self.rear_right_motor_.setPosition(float("inf"))

        self.front_left_motor_.setVelocity(0.0)
        self.front_right_motor_.setVelocity(0.0)
        self.rear_left_motor_.setVelocity(0.0)
        self.rear_right_motor_.setVelocity(0.0)

        self.pose_x_ = 0.0
        self.pose_y_ = 0.0
        self.velocity_x_ = 0.0
        self.velocity_y_ = 0.0
        self.orientation_ = np.array([0.0, 0.0, 0.0, 0.0])
        self.prev_velocity_ = None
        self.last_bearing_ = 0.0
        self.last_time_ = None
        self.pose_theta_ = 0.0
        self.prev_time = None

        self.WHEEL_RADIUS = 0.015  # Radius of the wheels (m)
        self.WHEEL_BASE_WIDTH = 0.13  # Distance between left and right wheels
        self.WHEEL_BASE_LENGTH = 0.12  # Distance between front and rear wheels

        rclpy.init(args=None)

        self.__node = rclpy.create_node(
            f"raspbotv2_robot_driver_{self._robot_id}", namespace=f"RaspbotV2_{self._robot_id}"
        )

        self.imu_publisher_ = self.__node.create_publisher(Imu, "imu", 10)
        self.odom_publisher_ = self.__node.create_publisher(Odometry, "odom", 10)

        self.tf_broadcast_ = tf2_ros.TransformBroadcaster(self.__node)

        self.imu_timer_ = self.__node.create_timer(0.1, self.publish_imu_)
        self.odom_timer_ = self.__node.create_timer(0.05, self.publish_odom_)

        self.cmd_vel_subscription_ = self.__node.create_subscription(
            Twist, "cmd_vel", self.cmd_vel_callback_, 10
        )

        # Let supervisor set the robots pose
        self.pose_sub = self.__node.create_subscription(
            Odometry, "pose_update", self.pose_callback_, 10
        )

        self.__node.get_logger().info("Raspbotv2 robot driver initialized")

    def pose_callback_(self, msg):
        # Update the robot's pose based on the received message
        self.pose_x_ = msg.pose.pose.position.x
        self.pose_y_ = msg.pose.pose.position.y
        self.pose_theta_ = msg.pose.pose.orientation.z
        self.orientation_ = np.array(
            [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ]
        )

    def read_acceleromter_(self):
        acc_data = self.acceleromter_.getValues()
        return Vector3(x=acc_data[0], y=acc_data[1], z=acc_data[2])

    def read_gyroscope_(self):
        gyro_data = self.gyroscope_.getValues()
        return Vector3(x=gyro_data[0], y=gyro_data[1], z=gyro_data[2])

    def read_magnetometer_(self):
        mag_data = self.magnetometer_.getValues()

        # Compute bearing (angle from north)
        rad = math.atan2(mag_data[1], mag_data[0])  # atan2(y, x) gives correct heading
        bearing = rad * 180.0 / np.pi  # Convert to degrees
        bearing -= 90.0  # Adjust for Webots orientation
        if bearing < 0:
            bearing += 360.0  # Normalize to [0, 360]

        return math.radians(bearing)

    def calculate_mecanum_odometry(self, wheel_velocities, current_time):
        """
        Calculate robot odometry using Mecanum wheel kinematics

        Args:
            wheel_velocities (list): Velocities of [FL, FR, RL, RR] wheels
            current_time (rclpy.time.Time): Current ROS time

        Returns:
            tuple: (delta_x, delta_y, delta_theta)
        """
        # Unpack wheel velocities (Front Left, Front Right, Rear Left, Rear Right)
        fl_vel, fr_vel, rl_vel, rr_vel = wheel_velocities

        # Time delta
        if self.prev_time is None:
            self.prev_time = current_time
            return 0, 0

        dt = (current_time - self.prev_time).nanoseconds / 1e9
        self.prev_time = current_time

        if dt <= 0:
            return 0, 0

        # Mecanum wheel inverse kinematics
        # These constants can be adjusted based on your specific robot geometry
        k1 = 1 / (2 * (self.WHEEL_BASE_LENGTH + self.WHEEL_BASE_WIDTH))

        # Calculate linear and angular velocities
        vx = self.WHEEL_RADIUS * (fl_vel + fr_vel + rl_vel + rr_vel) / 4
        vy = self.WHEEL_RADIUS * (-fl_vel + fr_vel + rl_vel - rr_vel) / 4

        # Integrate to get pose changes
        delta_x = vx * math.cos(self.pose_theta_) - vy * math.sin(self.pose_theta_) * dt
        delta_y = vx * math.sin(self.pose_theta_) + vy * math.cos(self.pose_theta_) * dt

        SIMULATION_SCALE = 0.25

        return delta_x * SIMULATION_SCALE, delta_y * SIMULATION_SCALE

    def update_odometry(self, wheel_velocities, current_time):
        """
        Update robot pose based on wheel velocities

        Args:
            wheel_velocities (list): Velocities of [FL, FR, RL, RR] wheels
            current_time (rclpy.time.Time): Current ROS time
        """
        # Calculate odometry deltas
        delta_x, delta_y = self.calculate_mecanum_odometry(wheel_velocities, current_time)

        # Update pose
        self.pose_x_ += delta_x
        self.pose_y_ += delta_y
        self.pose_theta_ = -self.read_magnetometer_()

        return self.pose_x_, self.pose_y_, self.pose_theta_

    def publish_imu_(self):
        now = self.__node.get_clock().now()
        timestamp = now.to_msg()

        imu_msg = Imu()
        imu_msg.header.stamp = timestamp
        imu_msg.header.frame_id = "raspbotv2_imu"

        # Compute acceleration from velocity if previous data exists
        if self.prev_velocity_ is not None and self.prev_time is not None:
            dt = (now - self.prev_time).nanoseconds * 1e-9  # Convert ns to seconds
            if dt > 0:  # Avoid division by zero
                acceleration = (self.velocity_x_ - self.prev_velocity_) / dt
            else:
                acceleration = 0.0
        else:
            acceleration = 0.0  # No previous data available

        # Store current values for next iteration
        self.prev_velocity_ = self.velocity_x_
        self.prev_time = now

        # Fill IMU message
        imu_msg.linear_acceleration.x = acceleration
        imu_msg.linear_acceleration.y = 0.0
        imu_msg.linear_acceleration.z = 0.0  # Assuming no vertical acceleration
        imu_msg.angular_velocity = self.read_gyroscope_()

        # Optional: Add covariance
        imu_msg.linear_acceleration_covariance = [0.02, 0, 0, 0, 0.02, 0, 0, 0, 0.02]
        imu_msg.angular_velocity_covariance = [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]
        imu_msg.orientation_covariance = [-1.0] * 9  # Unknown orientation

        self.imu_publisher_.publish(imu_msg)

    def publish_odom_(self):
        current_time = self.__node.get_clock().now()

        # Read wheel speeds
        wheel_speeds = self.read_wheel_speeds_()

        # Update odometry
        x, y, theta = self.update_odometry(wheel_speeds, current_time)

        # Convert euler angle to quaternion
        quat = R.from_euler('z', theta).as_quat()

        # Retrieve robot namespace (assume it's set as a ROS 2 parameter or node name)
        robot_namespace = self.__node.get_namespace().strip("/")
        if robot_namespace == "":
            robot_namespace = "default_robot"  # Fallback if no namespace is set

        # Prepare Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = f"{robot_namespace}/odom"  # Unique odom frame
        odom_msg.child_frame_id = f"{robot_namespace}/base_link"  # Unique base link

        # Position
        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.position.z = 0.0

        # Orientation
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]

        # Velocities (computed from wheel speeds)
        odom_msg.twist.twist.linear.x = (
            self.WHEEL_RADIUS
            * (wheel_speeds[0] + wheel_speeds[1] + wheel_speeds[2] + wheel_speeds[3])
            / 4
        )
        odom_msg.twist.twist.linear.y = (
            self.WHEEL_RADIUS
            * (-wheel_speeds[0] + wheel_speeds[1] + wheel_speeds[2] - wheel_speeds[3])
            / 4
        )
        odom_msg.twist.twist.angular.z = self.compute_angular_velocity(wheel_speeds)

        # Publish to a namespaced topic
        self.odom_publisher_.publish(odom_msg)

        # Publish TF transform
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = f"{robot_namespace}/odom"  # Unique odom frame
        t.child_frame_id = f"{robot_namespace}/base_link"  # Unique base link

        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0

        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcast_.sendTransform(t)

    def cmd_vel_callback_(self, msg):
        self.velocity_x_ = msg.linear.x
        self.velocity_y_ = msg.linear.y
        self.angular_z_ = msg.angular.z  # Store angular velocity

        # Mecanum wheel kinematics
        L = 0.06  # Half the distance between wheels (m)

        # Compute wheel velocities
        V1 = self.velocity_x_ - self.velocity_y_ - self.angular_z_ * L
        V2 = self.velocity_x_ + self.velocity_y_ + self.angular_z_ * L
        V3 = self.velocity_x_ + self.velocity_y_ - self.angular_z_ * L
        V4 = self.velocity_x_ - self.velocity_y_ + self.angular_z_ * L

        # Set wheel speeds in Webots
        self.front_left_motor_.setVelocity(min(V1, 10))
        self.front_right_motor_.setVelocity(min(V2, 10))
        self.rear_left_motor_.setVelocity(min(V3, 10))
        self.rear_right_motor_.setVelocity(min(V4, 10))

    def read_wheel_speeds_(self):
        return [
            self.front_left_motor_.getVelocity(),
            self.front_right_motor_.getVelocity(),
            self.rear_left_motor_.getVelocity(),
            self.rear_right_motor_.getVelocity(),
        ]

    def compute_angular_velocity(self, wheel_speeds):
        """
        Compute angular velocity from wheel speeds

        Args:
            wheel_speeds (list): Velocities of [FL, FR, RL, RR] wheels

        Returns:
            float: Angular velocity
        """
        k1 = 1 / (2 * (self.WHEEL_BASE_LENGTH + self.WHEEL_BASE_WIDTH))
        fl_vel, fr_vel, rl_vel, rr_vel = wheel_speeds
        return self.WHEEL_RADIUS * k1 * (-fl_vel + fr_vel - rl_vel + rr_vel)

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)
