import rclpy
from rclpy.node import Node
import rclpy.time
from sensor_msgs.msg import Image, LaserScan, CameraInfo
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation
import tf2_geometry_msgs


class ArucoNavNode(Node):
    def __init__(self):
        super().__init__('aruco_nav_node', namespace='aruco_nav')

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

        # Aruco Parameters
        self._aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self._aruco_params = cv2.aruco.DetectorParameters_create()

        # Marker Size (meters) – Set to actual printed marker size
        self._marker_size = 0.1  # 10 cm

        # Camera Matrix and Distortion Coefficients
        self._camera_matrix = None
        self._dist_coeffs = None
        self._latest_camera_transform = None

        self._cv_bridge = CvBridge()
        self.get_logger().info('Aruco Nav Node has been started')

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
        pass


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
