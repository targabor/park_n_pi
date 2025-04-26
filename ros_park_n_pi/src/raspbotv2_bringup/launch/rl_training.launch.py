import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher
from webots_ros2_driver.webots_controller import WebotsController
from launch.actions import SetEnvironmentVariable
import shutil


def modify_webots_world(context):
    """Generate a modified Webots world file with the specified number of robots."""
    num_robots = int(LaunchConfiguration('num_robots').perform(context))

    package_share_dir = get_package_share_directory("raspbotv2_bringup")
    original_world = os.path.join(package_share_dir, "webots_worlds", "rl_world.wbt")
    modified_world = os.path.join(package_share_dir, "webots_worlds", "rl_world_modified.wbt")

    # Copy the original world to a new file
    shutil.copyfile(original_world, modified_world)

    position = 0.0, 0.0, 0.13
    # Append robots to the copied world file
    with open(modified_world, 'a') as f:
        for i in range(1, num_robots + 1):
            f.write(
                f"""\nDEF RaspbotV2_{i} RaspbotV2 {{
                name "RaspbotV2_{i}"
                translation {position[0]} {position[1]} {position[2]}
                controller "<extern>"
            }}"""
            )
            position = position[0] + 0.3, position[1], position[2]

    return []  # No launch actions needed, just modifying the file


def spawn_robots(context):
    """Dynamically create robot nodes based on num_robots."""
    num_robots = int(LaunchConfiguration('num_robots').perform(context))

    package_share_dir = get_package_share_directory("raspbotv2_bringup")
    urdf_base_path = os.path.join(package_share_dir, "urdf", "ROBOT.urdf")

    robot_nodes = []
    for i in range(1, num_robots + 1):
        robot_name = f"RaspbotV2_{i}"

        # Modify URDF for each robot
        with open(urdf_base_path, 'r') as f:
            urdf = f.read().replace("ROBOT", robot_name)

        urdf_path = os.path.join(package_share_dir, "urdf", f"{robot_name}.urdf")
        with open(urdf_path, 'w') as f:
            f.write(urdf)

        # Append each robot node directly
        robot_nodes.append(
            WebotsController(
                robot_name=robot_name, parameters=[{"robot_description": urdf_path, "robot_id": i}]
            )
        )

    return robot_nodes  # Directly return the list of nodes


def generate_launch_description():
    package_share_dir = get_package_share_directory("raspbotv2_bringup")
    urdf_base_path = os.path.join(package_share_dir, "urdf", "SUPERVISOR.urdf")

    set_webots_env = [
        SetEnvironmentVariable('WEBOTS_HOME', '/usr/local/webots'),
        SetEnvironmentVariable(
            'PYTHONPATH',
            '/usr/local/webots/lib/controller/python:' + os.environ.get('PYTHONPATH', ''),
        ),
        SetEnvironmentVariable(
            'LD_LIBRARY_PATH',
            '/usr/local/webots/lib/controller:' + os.environ.get('LD_LIBRARY_PATH', ''),
        ),
    ]

    # Declare launch arguments
    num_robots_arg = DeclareLaunchArgument(
        'num_robots', default_value='8', description='Number of robots to spawn in the simulation'
    )

    # Webots world file
    webots_world = os.path.join(package_share_dir, "webots_worlds", "rl_world_modified.wbt")

    # Modify the Webots world at runtime
    modify_world_action = OpaqueFunction(function=modify_webots_world)

    # Webots launcher
    webots = WebotsLauncher(world=webots_world, ros2_supervisor=True)

    supervisor_node = WebotsController(
        robot_name='SUPERVISOR', parameters=[{"robot_description": urdf_base_path}]
    )

    # Spawn robots dynamically
    spawn_robots_action = OpaqueFunction(function=spawn_robots)
    return LaunchDescription(
        [num_robots_arg]
        + set_webots_env
        + [modify_world_action, webots, webots._supervisor, spawn_robots_action, supervisor_node]
    )
