from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import LifecycleNode, Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = PathJoinSubstitution([
        FindPackageShare("ugv_teb_bringup"),
        "config",
        "teb_controller_test.yaml",
    ])

    return LaunchDescription([
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="map_to_odom_static_tf",
            arguments=[
                "--x", "0",
                "--y", "0",
                "--z", "0",
                "--roll", "0",
                "--pitch", "0",
                "--yaw", "0",
                "--frame-id", "map",
                "--child-frame-id", "odom",
            ],
            output="screen",
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="odom_to_base_link_static_tf",
            arguments=[
                "--x", "0",
                "--y", "0",
                "--z", "0",
                "--roll", "0",
                "--pitch", "0",
                "--yaw", "0",
                "--frame-id", "odom",
                "--child-frame-id", "base_link",
            ],
            output="screen",
        ),

        LifecycleNode(
            package="nav2_controller",
            executable="controller_server",
            name="controller_server",
            namespace="",
            output="screen",
            parameters=[params_file],
        ),
    ])