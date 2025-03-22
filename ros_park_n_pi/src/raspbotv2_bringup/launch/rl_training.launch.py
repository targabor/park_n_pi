import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController
import shutil


def modify_webots_world(context):
    """Generate a modified Webots world file with the specified number of robots."""
    num_robots = int(LaunchConfiguration('num_robots').perform(context))

    package_share_dir = get_package_share_directory("raspbotv2_bringup")
    original_world = os.path.join(package_share_dir, "webots_worlds", "rl_world.wbt")
    modified_world = os.path.join(package_share_dir, "webots_worlds", "rl_world_modified.wbt")

    # Copy the original world to a new file
    shutil.copyfile(original_world, modified_world)

    # Append robots to the copied world file
    with open(modified_world, 'a') as f:
        for i in range(num_robots):
            f.write(
                f"""\nRaspbotV2 {{
                name "RaspbotV2_{i}"
                controller "<extern>"
            }}"""
            )

    return []  # No launch actions needed, just modifying the file


def spawn_robots(context):
    """Dynamically create robot nodes based on num_robots."""
    num_robots = int(LaunchConfiguration('num_robots').perform(context))

    package_share_dir = get_package_share_directory("raspbotv2_bringup")
    urdf_path = os.path.join(package_share_dir, "urdf", "ROBOT.urdf")

    robot_nodes = []
    for i in range(num_robots):
        robot_name = f"RaspbotV2_{i}"
        # Open the URDF file and replace the robot name
        with open(urdf_path, 'r') as f:
            urdf = f.read().replace("ROBOT", robot_name)
        # Write the modified URDF to a new file
        urdf_path = os.path.join(package_share_dir, "urdf", f"{robot_name}.urdf")
        with open(urdf_path, 'w') as f:
            f.write(urdf)

        robot_nodes.append(
            GroupAction(
                [
                    WebotsController(
                        robot_name=robot_name,
                        parameters=[{"robot_description": urdf_path, "robot_id": i}],
                    ),
                    Node(
                        package='webots_py_ros2_driver',
                        executable='robot_zmq_interface',
                        name=f'robot_rl_interface_{i}',
                        parameters=[
                            {'robot_id': i, 'coordinator_address': 'tcp://localhost:5556'}
                        ],
                    ),
                ]
            )
        )

    return robot_nodes


def generate_launch_description():
    package_share_dir = get_package_share_directory("raspbotv2_bringup")

    # Declare launch argument
    num_robots_arg = DeclareLaunchArgument(
        'num_robots', default_value='4', description='Number of robots to spawn in the simulation'
    )

    # Webots world file
    webots_world = os.path.join(package_share_dir, "webots_worlds", "rl_world_modified.wbt")

    # Modify the Webots world at runtime
    modify_world_action = OpaqueFunction(function=modify_webots_world)

    # Webots launcher
    webots = WebotsLauncher(world=webots_world)

    # Spawn robots dynamically
    spawn_robots_action = OpaqueFunction(function=spawn_robots)

    return LaunchDescription([num_robots_arg, modify_world_action, webots, spawn_robots_action])
