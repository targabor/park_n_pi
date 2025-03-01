import rclpy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3, Twist, TransformStamped
from nav_msgs.msg import Odometry
import numpy as np
import math
import tf2_ros


class Raspbotv2RobotDriver:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot
        timestep = int(self.__robot.getBasicTimeStep())

        self.acceleromter_ = self.__robot.getDevice("accelerometer")
        self.acceleromter_.enable(timestep)

        self.gyroscope_ = self.__robot.getDevice("gyroscope")
        self.gyroscope_.enable(timestep)

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
        self.theta_ = 0.0

        self.last_time_ = None

        rclpy.init(args=None)
        self.__node = rclpy.create_node("raspbotv2_robot_driver", namespace="RaspbotV2")

        self.imu_publisher_ = self.__node.create_publisher(Imu, "imu", 10)
        self.odom_publisher_ = self.__node.create_publisher(Odometry, "odom", 10)

        self.tf_broadcast_ = tf2_ros.TransformBroadcaster(self.__node)

        self.imu_timer_ = self.__node.create_timer(0.1, self.publish_imu_)
        self.odom_timer_ = self.__node.create_timer(0.05, self.publish_odom_)

        self.cmd_vel_subscription_ = self.__node.create_subscription(
            Twist, "cmd_vel", self.cmd_vel_callback_, 10
        )

        self.__node.get_logger().info("Raspbotv2 robot driver initialized")

    def read_acceleromter_(self):
        acc_data = self.acceleromter_.getValues()
        return Vector3(x=acc_data[0], y=acc_data[1], z=acc_data[2])

    def read_gyroscope_(self):
        gyro_data = self.gyroscope_.getValues()
        return Vector3(x=gyro_data[0], y=gyro_data[1], z=gyro_data[2])

    def publish_imu_(self):
        imu_msg = Imu()
        imu_msg.header.stamp = self.__node.get_clock().now().to_msg()
        imu_msg.header.frame_id = "raspbotv2_imu"
        imu_msg.linear_acceleration = self.read_acceleromter_()
        imu_msg.angular_velocity = self.read_gyroscope_()
        self.imu_publisher_.publish(imu_msg)

    def publish_odom_(self):
        current_time = self.__node.get_clock().now()

        if self.last_time_ is None:
            self.last_time_ = current_time
            return

        dt = (current_time - self.last_time_).nanoseconds / 1e9
        self.last_time_ = current_time

        if dt <= 0:
            return

        # Read IMU gyroscope data (angular velocity)
        gyro = self.read_gyroscope_()
        yaw_rate = gyro.z  # Angular velocity in rad/s
        delta_yaw = yaw_rate * dt

        # Update heading (theta_)
        self.theta_ += delta_yaw

        # Convert heading to quaternion
        qw = math.cos(self.theta_ / 2)
        qz = math.sin(self.theta_ / 2)
        self.orientation_ = np.array([0.0, 0.0, qz, qw])

        # Integrate velocity to compute new position
        delta_x = (
            self.velocity_x_ * math.cos(self.theta_) - self.velocity_y_ * math.sin(self.theta_)
        ) * dt
        delta_y = (
            self.velocity_x_ * math.sin(self.theta_) + self.velocity_y_ * math.cos(self.theta_)
        ) * dt

        self.pose_x_ += delta_x
        self.pose_y_ += delta_y

        # Prepare Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        odom_msg.pose.pose.position.x = self.pose_x_
        odom_msg.pose.pose.position.y = self.pose_y_
        odom_msg.pose.pose.position.z = 0.0

        odom_msg.pose.pose.orientation.x = self.orientation_[0]
        odom_msg.pose.pose.orientation.y = self.orientation_[1]
        odom_msg.pose.pose.orientation.z = self.orientation_[2]
        odom_msg.pose.pose.orientation.w = self.orientation_[3]

        odom_msg.twist.twist.linear.x = self.velocity_x_
        odom_msg.twist.twist.linear.y = self.velocity_y_
        odom_msg.twist.twist.angular.z = yaw_rate

        self.odom_publisher_.publish(odom_msg)

        # Publish TF transform
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"

        t.transform.translation.x = self.pose_x_
        t.transform.translation.y = self.pose_y_
        t.transform.translation.z = 0.0

        t.transform.rotation.x = self.orientation_[0]
        t.transform.rotation.y = self.orientation_[1]
        t.transform.rotation.z = self.orientation_[2]
        t.transform.rotation.w = self.orientation_[3]

        self.tf_broadcast_.sendTransform(t)

    def cmd_vel_callback_(self, msg):
        self.velocity_x_ = msg.linear.x
        self.velocity_y_ = msg.linear.y
        self.angular_z_ = msg.angular.z  # Store angular velocity

        # Mecanum wheel kinematics
        L = 0.082915  # Half the distance between wheels (m)

        # Compute wheel velocities
        V1 = self.velocity_x_ - self.velocity_y_ - self.angular_z_ * L
        V2 = self.velocity_x_ + self.velocity_y_ + self.angular_z_ * L
        V3 = self.velocity_x_ + self.velocity_y_ - self.angular_z_ * L
        V4 = self.velocity_x_ - self.velocity_y_ + self.angular_z_ * L

        # Set wheel speeds in Webots
        self.front_left_motor_.setVelocity(V1)
        self.front_right_motor_.setVelocity(V2)
        self.rear_left_motor_.setVelocity(V3)
        self.rear_right_motor_.setVelocity(V4)

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)
