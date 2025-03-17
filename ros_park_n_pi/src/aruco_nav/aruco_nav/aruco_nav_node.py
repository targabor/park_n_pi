import rclpy
from rclpy.node import Node
import rclpy.time
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation
import tf2_geometry_msgs
import random
import time
from std_srvs.srv import Trigger
from rclpy.callback_groups import ReentrantCallbackGroup


class ArucoNavNode(Node):
    def __init__(self):
        super().__init__('aruco_nav_node', namespace='aruco_nav')
        # Create a callback group that allows concurrent callbacks
        self._callback_group = ReentrantCallbackGroup()
        # Subscribers
        self._camera_sub = self.create_subscription(
            Image, '/RaspbotV2/front_camera/image_color', self.image_callback, 10
        )
        self._laser_scan_sub = self.create_subscription(
            LaserScan, '/RaspbotV2/top_lidar', self.laser_callback, 10
        )
        self._camera_info_sub = self.create_subscription(
            CameraInfo, '/RaspbotV2/front_camera/camera_info', self.camera_info_callback, 10
        )
        self._odom_sub = self.create_subscription(
            Odometry, '/RaspbotV2/odom', self._odom_callback, 10
        )

        # Publishers
        self._cmd_vel_pub = self.create_publisher(Twist, '/RaspbotV2/cmd_vel', 10)
        self._aruco_img_pub = self.create_publisher(Image, 'aruco_image', 10)

        # TF Broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self._tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=5.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # Timer
        self.create_timer(0.1, self._update_camera_transform)

        # Service
        self._search_service = self.create_service(
            Trigger,
            'start_marker_search',
            self._search_service_callback,
            callback_group=self._callback_group,
        )

        # Aruco Parameters
        self._aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self._aruco_params = cv2.aruco.DetectorParameters_create()

        # Marker Size (meters) – Set to actual printed marker size
        self._marker_size = 0.1  # 10 cm

        # Camera Matrix and Distortion Coefficients
        self._camera_matrix = None
        self._dist_coeffs = None
        self._latest_camera_transform = None

        # Variables for sensor data
        self._laser_scan = None
        self._twist = Twist()

        # Variables to navigate
        self._aruco_found = False
        self._desired_distance = 0.3
        self._max_speed = 1.0
        self._max_turn_speed = 3.5
        self._kp = 1.0
        self._kd = 0.1
        self._follow_right = True

        self._prev_error = 0.0

        self.current_speed = 0.0

        self._current_yaw = 0.0

        self._cv_bridge = CvBridge()
        self.get_logger().info('Aruco Nav Node has been started')

    def _odom_callback(self, msg):
        # Extract yaw from quaternion
        orientation = msg.pose.pose.orientation
        quat = np.array([orientation.x, orientation.y, orientation.z, orientation.w])

        # Convert quaternion to euler angles
        r = Rotation.from_quat(quat)
        yaw = r.as_euler('zyx', degrees=False)[0]  # Extract yaw in radians

        # Store current yaw
        self._current_yaw = yaw
        self.get_logger().info(f"Updated Yaw: {self._current_yaw}")  # Debug print

    def _update_camera_transform(self):
        """Continuously updates the latest front_camera -> map transform"""
        try:
            self._latest_camera_transform = self._tf_buffer.lookup_transform(
                "map",
                "front_camera",
                time=rclpy.time.Time(seconds=0),  # Get latest available transform
                timeout=rclpy.duration.Duration(seconds=1.0),  # Wait up to 1 sec if unavailable
            )
        except Exception as e:
            self.get_logger().warn(f"Could not get transform map -> front_camera: {e}")

    def camera_info_callback(self, msg):
        """Reads camera calibration parameters and unsubscribes after first message."""
        if self._camera_matrix is None:
            self._camera_matrix = np.array(msg.k).reshape(3, 3)
            self._dist_coeffs = np.array(msg.d)

            self.get_logger().info(
                'Camera calibration data received. Unsubscribing from /camera_info.'
            )

            # Unsubscribe since we only need this once
            if self._camera_info_sub is not None:
                self.destroy_subscription(self._camera_info_sub)
                self._camera_info_sub = None

    def image_callback(self, msg):
        if self._camera_matrix is None:
            self.get_logger().warn(
                'Camera calibration data not available. Skipping image processing.'
            )
            return

        # Convert ROS Image to OpenCV Image
        cv_image = self._cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        # Convert to Grayscale
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # Detect Aruco Markers
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray_image, self._aruco_dict, parameters=self._aruco_params
        )
        if ids is not None:

            # Estimate Pose
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self._marker_size, self._camera_matrix, self._dist_coeffs
            )

            # Publish image with markers
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            self._aruco_img_pub.publish(self._cv_bridge.cv2_to_imgmsg(cv_image, 'bgr8'))

            # Publish TF
            for i, marker_id in enumerate(ids.flatten()):
                self.publish_tf(marker_id, tvecs[i], rvecs[i])

    def publish_tf(self, marker_id, tvec, rvec):
        """Publish ArUco marker pose as TF transform relative to map frame"""

        if self._latest_camera_transform is None:
            self.get_logger().warn("No camera transform available. Skipping TF publish.")
            return

        # Create the transform relative to front_camera
        marker_pose = tf2_geometry_msgs.PoseStamped()
        marker_pose.header.frame_id = "front_camera"
        marker_pose.header.stamp = self.get_clock().now().to_msg()
        marker_pose.pose.position.x = float(tvec[0][0])
        marker_pose.pose.position.y = float(tvec[0][1])
        marker_pose.pose.position.z = float(tvec[0][2])

        # Convert rotation vector to quaternion
        rot_matrix = np.eye(4)
        rot_matrix[:3, :3] = cv2.Rodrigues(rvec)[0]
        quat = self.quaternion_from_matrix(rot_matrix)

        marker_pose.pose.orientation.x = float(quat[0])
        marker_pose.pose.orientation.y = float(quat[1])
        marker_pose.pose.orientation.z = float(quat[2])
        marker_pose.pose.orientation.w = float(quat[3])

        # Transform marker pose into map frame
        try:
            marker_pose_map = tf2_geometry_msgs.do_transform_pose(
                marker_pose.pose, self._latest_camera_transform
            )

            # Publish the marker relative to map
            final_transform = TransformStamped()
            final_transform.header.stamp = self.get_clock().now().to_msg()
            final_transform.header.frame_id = "map"
            final_transform.child_frame_id = f"aruco_marker_{marker_id}"
            # Corrected: Extract position and orientation from the transformed pose
            final_transform.transform.translation.x = marker_pose_map.position.x
            final_transform.transform.translation.y = marker_pose_map.position.y
            final_transform.transform.translation.z = marker_pose_map.position.z

            final_transform.transform.rotation = marker_pose_map.orientation

            self.tf_static_broadcaster.sendTransform(final_transform)
            self._aruco_found = True
            self.get_logger().info(f"Published ArUco marker {marker_id} in map frame")

        except Exception as e:
            self.get_logger().warn(f"Failed to transform marker {marker_id} to map frame: {e}")

    def quaternion_from_matrix(self, matrix):
        r = Rotation.from_matrix(matrix[:3, :3])
        return r.as_quat()

    def quaternion_matrix(self, quaternion):
        r = Rotation.from_quat(quaternion)
        mat = np.eye(4)
        mat[:3, :3] = r.as_matrix()
        return mat

    def laser_callback(self, msg):
        self._laser_scan = msg

    def _search_service_callback(self, request, response):
        """Handle service request to start marker search."""
        # Check if already searching
        if (
            hasattr(self, '_walk_timer')
            and self._walk_timer is not None
            and self._walk_timer.is_ready()
        ):
            response.success = False
            response.message = "Search already in progress."
            return response

        self.get_logger().info("Starting marker search.")

        # Initialize state variables
        self._aruco_found = False
        self._last_scan_time = time.time()
        self._stuck_counter = 0
        self._max_stuck_attempts = 3
        self._in_360_scan = False
        self._scan_current_yaw = None
        self._scan_rotated_angle = 0.0

        # Start the timer for periodic execution
        self._walk_timer = self.create_timer(0.05, self._random_walk_step)

        response.success = True
        response.message = "Marker search started successfully."
        return response

    def _random_walk_step(self):
        """Execute one step of the random walk."""
        # Check if we're done
        if self._aruco_found:
            self._cleanup_and_finish()
            return

        # Process based on current state
        if self._in_360_scan:
            self._execute_360_scan()
        else:
            self._execute_normal_movement()

        # Publish velocity command
        self._cmd_vel_pub.publish(self._twist)

    def _execute_normal_movement(self):
        """Execute normal movement and obstacle avoidance."""
        # Process laser scan
        ranges, angles = self._process_laser_scan()

        # Detect obstacles in front
        front_dist = self._detect_front_obstacles(ranges, angles)

        # Obstacle avoidance logic
        if front_dist < self._desired_distance:
            self._handle_obstacle(front_dist)
        else:
            self._move_normally()

        # Check if it's time for a 360-degree scan
        if time.time() - self._last_scan_time > 10:  # Every 10 seconds
            self._start_360_scan()

    def _process_laser_scan(self):
        """Process laser scan data."""
        ranges = np.array(self._laser_scan.ranges)
        ranges = np.clip(ranges, self._laser_scan.range_min, self._laser_scan.range_max)

        angles = (
            self._laser_scan.angle_min + np.arange(len(ranges)) * self._laser_scan.angle_increment
        )

        return ranges, angles

    def _detect_front_obstacles(self, ranges, angles):
        """Detect obstacles in front of the robot."""
        front_indices = np.where(abs(angles) < 0.3)[0]  # Looking at ±0.3 rad (~±17°)
        if len(front_indices) > 0:
            return np.min(ranges[front_indices])
        else:
            return self._laser_scan.range_max  # Assume no obstacle if empty

    def _handle_obstacle(self, front_dist):
        """Handle obstacle avoidance."""
        self._stuck_counter += 1
        turn_direction = random.choice([-1, 1])  # Randomly turn left or right

        if self._stuck_counter >= self._max_stuck_attempts:
            self._execute_stuck_escape(turn_direction)
        else:
            # Normal obstacle avoidance
            self._twist.linear.x = 0.0
            self._twist.angular.z = turn_direction * self._max_turn_speed

    def _execute_stuck_escape(self, turn_direction):
        """Execute escape maneuver when stuck."""
        # This is a case where we need to create a new temporary timer
        # to sequence the escape maneuver without blocking
        self._walk_timer.cancel()  # Pause the main timer

        # First move backward
        self._twist.linear.x = -0.2
        self._twist.angular.z = 0.0
        self._cmd_vel_pub.publish(self._twist)

        # Create a timer for the next step
        self._stuck_timer = self.create_timer(
            1.0, lambda: self._execute_stuck_turn(turn_direction)  # 1 second delay
        )
        self._stuck_timer.cancel_on_destroy = True

    def _execute_stuck_turn(self, turn_direction):
        """Execute turn after backing up to escape being stuck."""
        # Cancel the single-shot timer
        self._stuck_timer.cancel()

        # Start turning
        self._twist.linear.x = 0.0
        self._twist.angular.z = turn_direction * self._max_turn_speed
        self._cmd_vel_pub.publish(self._twist)

        # Create a timer to resume normal operation
        self._stuck_timer = self.create_timer(
            2.0, self._resume_after_stuck  # 2 seconds of turning
        )
        self._stuck_timer.cancel_on_destroy = True

    def _resume_after_stuck(self):
        """Resume normal operation after stuck escape sequence."""
        self._stuck_timer.cancel()
        self._stuck_counter = 0  # Reset stuck counter
        self._walk_timer = self.create_timer(0.05, self._random_walk_step)  # Resume main timer

    def _move_normally(self):
        """Execute normal movement."""
        self._twist.linear.x = self._max_speed
        self._twist.angular.z = random.uniform(-0.3, 0.3)  # Small drift for randomness
        self._stuck_counter = 0  # Reset counter since we're moving fine

    def _start_360_scan(self):
        """Start a 360-degree scan."""
        self.get_logger().info("Performing 360-degree scan")
        self._twist.linear.x = 0.0  # Stop movement
        self._cmd_vel_pub.publish(self._twist)

        # Create a timer to start rotation after stopping
        self._walk_timer.cancel()  # Pause the main timer
        self._scan_prep_timer = self.create_timer(1.0, self._begin_360_rotation)  # 1 second delay
        self._scan_prep_timer.cancel_on_destroy = True

    def _begin_360_rotation(self):
        """Begin the 360-degree rotation."""
        self._scan_prep_timer.cancel()

        # Start 360 scan state
        self._in_360_scan = True
        self._scan_current_yaw = self._current_yaw
        self._scan_rotated_angle = 0.0

        # Configure for rotation
        self._twist.linear.x = 0.0
        self._twist.angular.z = -self._max_turn_speed
        self._cmd_vel_pub.publish(self._twist)

        # Resume the main timer to handle the rotation
        self._walk_timer = self.create_timer(0.05, self._random_walk_step)

    def _execute_360_scan(self):
        """Execute one step of the 360-degree scan."""

        # Define angle_diff function
        def angle_diff(a, b):
            return (a - b + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]

        # Check if we've completed the scan
        if self._scan_rotated_angle >= 2 * np.pi:
            self._in_360_scan = False
            self._last_scan_time = time.time()  # Reset scan timer
            return

        # Calculate incremental rotation
        delta_yaw = angle_diff(self._current_yaw, self._scan_current_yaw)
        self._scan_rotated_angle += abs(delta_yaw)  # Accumulate rotation
        self._scan_current_yaw = self._current_yaw  # Update for next iteration

        # Log progress
        self.get_logger().info(
            f"Rotated: {self._scan_rotated_angle:.2f} rad, Current yaw: {self._current_yaw:.2f}"
        )

        # Ensure we're still rotating
        self._twist.linear.x = 0.0
        self._twist.angular.z = -self._max_turn_speed

    def _cleanup_and_finish(self):
        """Clean up when marker is found."""
        if hasattr(self, '_walk_timer') and self._walk_timer is not None:
            self._walk_timer.cancel()
            self._walk_timer = None

        # Stop the robot
        self._twist.linear.x = 0.0
        self._twist.angular.z = 0.0
        self._cmd_vel_pub.publish(self._twist)

        self.get_logger().info("Marker found. Search completed.")


def main(args=None):
    rclpy.init(args=args)

    node = ArucoNavNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by keyboard interrupt')
    except Exception as e:
        node.get_logger().error(f'Exception in node: {str(e)}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
