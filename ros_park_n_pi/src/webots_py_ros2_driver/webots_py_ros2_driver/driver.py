import rclpy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
import numpy as np
import math


class Raspbotv2RobotDriver:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot

        self.acceleromter_ = self.__robot.getDevice("accelerometer")
        self.acceleromter_.enable(100)

        self.gyroscope_ = self.__robot.getDevice("gyroscope")
        self.gyroscope_.enable(100)

        self.pose_x_ = 0.0
        self.pose_y_ = 0.0
        self.velocity_x_ = 0.0
        self.velocity_y_ = 0.0
        self.orientation_ = np.array([0.0, 0.0, 0.0, 0.0])

        rclpy.init(args=None)
        self.__node = rclpy.create_node("raspbotv2_robot_driver", namespace="RaspbotV2")

        self.imu_publisher_ = self.__node.create_publisher(Imu, "imu", 10)
        self.odom_publisher_ = self.__node.create_publisher(Odometry, "odom", 10)

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
        # self.__node.get_logger().info(f"Publishing IMU data: {imu_msg}")
        self.imu_publisher_.publish(imu_msg)

    def publish_odom_(self):
        odom_msg = Odometry()

        # Read sensor data
        accel_data = self.read_acceleromter_()
        gyro_data = self.read_gyroscope_()

        accel_x, accel_y, accel_z = accel_data.x, accel_data.y, accel_data.z
        gyro_x, gyro_y, gyro_z = gyro_data.x, gyro_data.y, gyro_data.z
        dt = 0.01

        # Update orientation using gyroscope data
        # Convert angular velocity to rotation increment
        delta_roll = gyro_x * dt
        delta_pitch = gyro_y * dt
        delta_yaw = gyro_z * dt

        # Update quaternion - simplified approach using small-angle approximation
        self.orientation_[0] += 0.5 * (
            self.orientation_[3] * delta_roll
            - self.orientation_[2] * delta_pitch
            + self.orientation_[1] * delta_yaw
        )
        self.orientation_[1] += 0.5 * (
            self.orientation_[2] * delta_roll
            + self.orientation_[3] * delta_pitch
            - self.orientation_[0] * delta_yaw
        )
        self.orientation_[2] += 0.5 * (
            -self.orientation_[1] * delta_roll
            + self.orientation_[0] * delta_pitch
            + self.orientation_[3] * delta_yaw
        )
        self.orientation_[3] += 0.5 * (
            -self.orientation_[0] * delta_roll
            - self.orientation_[1] * delta_pitch
            - self.orientation_[2] * delta_yaw
        )

        # Normalize quaternion to prevent drift
        quat_magnitude = (
            self.orientation_[0] ** 2
            + self.orientation_[1] ** 2
            + self.orientation_[2] ** 2
            + self.orientation_[3] ** 2
        ) ** 0.5
        self.orientation_ = [q / quat_magnitude for q in self.orientation_]

        # Transform acceleration from sensor frame to world frame
        # This is a simplified version - full implementation would use quaternion rotation
        # Get yaw angle from quaternion (simplified)
        yaw = 2 * math.atan2(self.orientation_[2], self.orientation_[3])

        # Rotate acceleration to world frame
        world_accel_x = accel_x * math.cos(yaw) - accel_y * math.sin(yaw)
        world_accel_y = accel_x * math.sin(yaw) + accel_y * math.cos(yaw)

        # Integrate acceleration to get velocity (remove gravity component if needed)
        self.velocity_x_ += world_accel_x * dt
        self.velocity_y_ += world_accel_y * dt

        # Integrate velocity to get position
        self.pose_x_ += self.velocity_x_ * dt
        self.pose_y_ += self.velocity_y_ * dt

        # Fill odometry message
        odom_msg.header.stamp = self.__node.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        # Position
        odom_msg.pose.pose.position.x = self.pose_x_
        odom_msg.pose.pose.position.y = self.pose_y_
        odom_msg.pose.pose.position.z = 0.0

        # Orientation (quaternion)
        odom_msg.pose.pose.orientation.x = self.orientation_[0]
        odom_msg.pose.pose.orientation.y = self.orientation_[1]
        odom_msg.pose.pose.orientation.z = self.orientation_[2]
        odom_msg.pose.pose.orientation.w = self.orientation_[3]

        # Velocity
        odom_msg.twist.twist.linear.x = self.velocity_x_
        odom_msg.twist.twist.linear.y = self.velocity_y_
        odom_msg.twist.twist.linear.z = 0.0

        # Angular velocity
        odom_msg.twist.twist.angular.x = gyro_x
        odom_msg.twist.twist.angular.y = gyro_y
        odom_msg.twist.twist.angular.z = gyro_z

        # Publish message
        self.odom_publisher_.publish(odom_msg)

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)
        self.publish_imu_()
        self.publish_odom_()
