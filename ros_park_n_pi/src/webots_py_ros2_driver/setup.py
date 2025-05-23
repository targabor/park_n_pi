from setuptools import find_packages, setup

package_name = "webots_py_ros2_driver"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="gabor",
    maintainer_email="tar.gabor14@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "xbox_teleop = webots_py_ros2_driver.xbox_teleop:main",
            "robot_zmq_interface = webots_py_ros2_driver.robot_zmq_interface:main",
        ]
    },
)
