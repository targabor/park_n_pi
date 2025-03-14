import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist


class XboxTeleopNode(Node):
    def __init__(self):
        super().__init__('xbox_teleop_node')
        self.cmd_publisher_ = self.create_publisher(Twist, 'RaspbotV2/cmd_vel', 10)
        self.joy_subscription_ = self.create_subscription(Joy, '/joy', self.joy_callback, 10)

        self.MAX_SPEED = 0.25
        self.MAX_ANGULAR_SPEED = 20

        self.get_logger().info("Xbox teleop node initialized")

    def joy_callback(self, msg: Joy):
        twist = Twist()

        left_stick_x = msg.axes[0]
        rt_trigger = (-msg.axes[5] + 1) / 2  # Rescale from [-1, 1] to [0, 1]
        lt_trigger = (-msg.axes[2] + 1) / 2  # Rescale from [-1, 1] to [0, 1]

        speed = rt_trigger * self.MAX_ANGULAR_SPEED  # RT increases speed
        brake = lt_trigger * self.MAX_ANGULAR_SPEED  # LT reduces speed
        twist.linear.x = speed - brake  # Apply throttle & brake

        # Steering
        twist.angular.z = left_stick_x * self.MAX_ANGULAR_SPEED  # Left stick for turning

        self.cmd_publisher_.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = XboxTeleopNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
