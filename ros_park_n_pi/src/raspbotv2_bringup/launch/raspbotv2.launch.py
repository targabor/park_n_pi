import launch
from launch import LaunchDescription
import launch.event_handlers
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController
from launch_ros.actions import Node
from launch.substitutions import Command
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    urdf_path = os.path.join(
        get_package_share_directory("raspbotv2_bringup"), "urdf", "RaspbotV2.urdf"
    )

    webots = WebotsLauncher(
        world=os.path.join(
            get_package_share_directory("raspbotv2_bringup"),
            "webots_worlds",
            "world.wbt",
        )
    )

    raspbotv2_robot_driver = WebotsController(
        robot_name="RaspbotV2", parameters=[{"robot_description": urdf_path}]
    )

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        parameters=[{"use_gui": False}],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{"robot_description": Command(["cat ", urdf_path])}],
    )

    return LaunchDescription(
        [
            webots,
            raspbotv2_robot_driver,
            joint_state_publisher,
            robot_state_publisher,
            launch.actions.RegisterEventHandler(
                event_handler=launch.event_handlers.OnProcessExit(
                    target_action=webots,
                    on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
                )
            ),
        ]
    )
