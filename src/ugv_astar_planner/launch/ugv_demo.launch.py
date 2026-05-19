from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")
    teb_config_file = LaunchConfiguration("teb_config_file")

    include_foxglove = LaunchConfiguration("include_foxglove")
    include_recorder = LaunchConfiguration("include_recorder")
    include_scenario_runner = LaunchConfiguration("include_scenario_runner")
    include_sac_adapter = LaunchConfiguration("include_sac_adapter")
    use_teb = LaunchConfiguration("use_teb")

    foxglove_address = LaunchConfiguration("foxglove_address")
    foxglove_port = LaunchConfiguration("foxglove_port")

    default_config = PathJoinSubstitution([
        FindPackageShare("ugv_astar_planner"),
        "config",
        "ugv_demo.yaml",
    ])

    default_teb_config = PathJoinSubstitution([
        FindPackageShare("ugv_astar_planner"),
        "config",
        "teb_controller.yaml",
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file",
            default_value=default_config,
            description="Path to YAML parameter file for demo nodes.",
        ),
        DeclareLaunchArgument(
            "teb_config_file",
            default_value=default_teb_config,
            description="Path to YAML parameter file for Nav2 TEB controller_server.",
        ),
        DeclareLaunchArgument(
            "include_foxglove",
            default_value="true",
            description="Whether to launch foxglove_bridge.",
        ),
        DeclareLaunchArgument(
            "include_recorder",
            default_value="true",
            description="Whether to record metrics CSV.",
        ),
        DeclareLaunchArgument(
            "include_scenario_runner",
            default_value="false",
            description="Whether to run automatic goal sequence.",
        ),
        DeclareLaunchArgument(
            "include_sac_adapter",
            default_value="true",
            description="Whether to run SAC adapter interface.",
        ),
        DeclareLaunchArgument(
            "use_teb",
            default_value="false",
            description="Use TEB controller instead of local_planner + pure_pursuit.",
        ),
        DeclareLaunchArgument(
            "foxglove_address",
            default_value="127.0.0.1",
            description="Foxglove bridge bind address.",
        ),
        DeclareLaunchArgument(
            "foxglove_port",
            default_value="8765",
            description="Foxglove bridge port.",
        ),

        ExecuteProcess(
            condition=IfCondition(include_foxglove),
            cmd=[
                "ros2",
                "launch",
                "foxglove_bridge",
                "foxglove_bridge_launch.xml",
                ["port:=", foxglove_port],
                ["address:=", foxglove_address],
            ],
            output="screen",
        ),

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
            package="ugv_astar_planner",
            executable="path_demo",
            name="astar_planner_demo",
            parameters=[config_file],
            output="screen",
        ),

        Node(
            package="ugv_astar_planner",
            executable="fake_base",
            name="fake_base",
            parameters=[config_file],
            output="screen",
        ),

        Node(
            condition=UnlessCondition(use_teb),
            package="ugv_astar_planner",
            executable="local_planner",
            name="local_trajectory_planner",
            parameters=[config_file],
            output="screen",
        ),

        Node(
            condition=UnlessCondition(use_teb),
            package="ugv_astar_planner",
            executable="pure_pursuit",
            name="pure_pursuit",
            parameters=[config_file],
            output="screen",
        ),

        Node(
            condition=IfCondition(use_teb),
            package="nav2_controller",
            executable="controller_server",
            name="controller_server",
            namespace="",
            parameters=[teb_config_file],
            remappings=[
                ("/cmd_vel", "/cmd_vel_raw"),
            ],
            output="screen",
        ),

        TimerAction(
            condition=IfCondition(use_teb),
            period=1.0,
            actions=[
                Node(
                    package="nav2_lifecycle_manager",
                    executable="lifecycle_manager",
                    name="lifecycle_manager_controller",
                    output="screen",
                    parameters=[
                        {
                            "use_sim_time": False,
                            "autostart": True,
                            "node_names": ["controller_server"],
                        }
                    ],
                ),
            ],
        ),

        Node(
            condition=IfCondition(use_teb),
            package="ugv_astar_planner",
            executable="teb_path_sender",
            name="teb_path_sender",
            parameters=[config_file],
            output="screen",
        ),

        Node(
            package="ugv_astar_planner",
            executable="safety_filter",
            name="safety_filter",
            parameters=[config_file],
            output="screen",
        ),

        Node(
            package="ugv_astar_planner",
            executable="metrics_monitor",
            name="metrics_monitor",
            output="screen",
        ),

        Node(
            condition=IfCondition(include_sac_adapter),
            package="ugv_astar_planner",
            executable="sac_adapter",
            name="sac_adapter",
            parameters=[config_file],
            output="screen",
        ),

        Node(
            condition=IfCondition(include_recorder),
            package="ugv_astar_planner",
            executable="experiment_recorder",
            name="experiment_recorder",
            parameters=[config_file],
            output="screen",
        ),

        Node(
            condition=IfCondition(include_scenario_runner),
            package="ugv_astar_planner",
            executable="scenario_runner",
            name="scenario_runner",
            parameters=[config_file],
            output="screen",
        ),
    ])
