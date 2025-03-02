import launch
from launch import LaunchDescription
import launch.event_handlers
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController
from launch_ros.actions import Node
from launch.substitutions import Command
import os
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    package_share_dir = get_package_share_directory("raspbotv2_bringup")

    urdf_path = os.path.join(package_share_dir, "urdf", "RaspbotV2.urdf")

    ekf_config_path = os.path.join(package_share_dir, "config", "ekf_config.yaml")

    slam_params_path = os.path.join(package_share_dir, "config", "mapper_params_online_async.yaml")

    webots = WebotsLauncher(world=os.path.join(package_share_dir, "webots_worlds", "world.wbt"))

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

    robot_localization = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        parameters=[ekf_config_path],
    )

    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("slam_toolbox"), "launch", "online_async_launch.py"
            )
        ),
        launch_arguments={"slam_params_file": slam_params_path, "use_sim_time": "false"}.items(),
    )

    joy_detecor = Node(package="joy", executable="joy_node", name="joy_node")

    xbox_teleop = Node(
        package="webots_py_ros2_driver", executable="xbox_teleop", name="xbox_teleop"
    )
    return LaunchDescription(
        [
            webots,
            raspbotv2_robot_driver,
            joint_state_publisher,
            robot_state_publisher,
            joy_detecor,
            xbox_teleop,
            slam_toolbox,
            launch.actions.RegisterEventHandler(
                event_handler=launch.event_handlers.OnProcessExit(
                    target_action=webots,
                    on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
                )
            ),
        ]
    )
