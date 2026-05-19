import json
import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import Float32, String
from visualization_msgs.msg import MarkerArray


class MetricsMonitor(Node):
    def __init__(self):
        super().__init__("metrics_monitor")

        self.local_path_timeout = 2.0

        self.create_subscription(Path, "/astar_path", self.on_global_path, 10)
        self.create_subscription(Path, "/local_trajectory", self.on_local_trajectory, 10)
        self.create_subscription(Path, "/local_plan", self.on_teb_local_plan, 10)
        self.create_subscription(PoseStamped, "/robot_pose", self.on_robot_pose, 10)
        self.create_subscription(PoseStamped, "/astar_goal", self.on_goal_pose, 10)
        self.create_subscription(Twist, "/cmd_vel_raw", self.on_cmd_raw, 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd_safe, 10)
        self.create_subscription(String, "/safety_status", self.on_safety_status, 10)
        self.create_subscription(MarkerArray, "/dynamic_obstacles", self.on_dynamic_obstacles, 10)

        self.goal_distance_pub = self.create_publisher(Float32, "/metrics/goal_distance", 10)
        self.global_path_length_pub = self.create_publisher(Float32, "/metrics/global_path_length", 10)
        self.local_path_length_pub = self.create_publisher(Float32, "/metrics/local_trajectory_length", 10)
        self.raw_linear_pub = self.create_publisher(Float32, "/metrics/cmd_raw_linear", 10)
        self.safe_linear_pub = self.create_publisher(Float32, "/metrics/cmd_safe_linear", 10)
        self.safe_angular_pub = self.create_publisher(Float32, "/metrics/cmd_safe_angular", 10)
        self.safety_scale_pub = self.create_publisher(Float32, "/metrics/safety_scale", 10)
        self.min_obstacle_pub = self.create_publisher(Float32, "/metrics/min_dynamic_obstacle_distance", 10)
        self.nav_status_pub = self.create_publisher(String, "/metrics/navigation_status", 10)
        self.local_path_source_pub = self.create_publisher(String, "/metrics/local_trajectory_source", 10)
        self.summary_pub = self.create_publisher(String, "/metrics/summary", 10)

        self.global_path = []
        self.local_trajectory_path = []
        self.teb_local_plan = []
        self.dynamic_obstacles = []

        self.last_local_trajectory_time = 0.0
        self.last_teb_local_plan_time = 0.0

        self.has_robot_pose = False
        self.has_goal = False

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.goal_x = 0.0
        self.goal_y = 0.0

        self.cmd_raw_linear = 0.0
        self.cmd_safe_linear = 0.0
        self.cmd_safe_angular = 0.0
        self.safety_status = "UNKNOWN"

        self.goal_tolerance = 0.35
        self.last_goal = None
        self.goal_start_time = time.time()
        self.reached_once = False

        self.timer = self.create_timer(0.2, self.publish_metrics)

        self.get_logger().info("Metrics monitor started.")

    def now_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def path_from_msg(self, msg):
        return [
            (pose.pose.position.x, pose.pose.position.y)
            for pose in msg.poses
        ]

    def on_global_path(self, msg):
        self.global_path = self.path_from_msg(msg)

    def on_local_trajectory(self, msg):
        self.local_trajectory_path = self.path_from_msg(msg)
        self.last_local_trajectory_time = self.now_seconds()

    def on_teb_local_plan(self, msg):
        self.teb_local_plan = self.path_from_msg(msg)
        self.last_teb_local_plan_time = self.now_seconds()

    def on_robot_pose(self, msg):
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y
        self.has_robot_pose = True

    def on_goal_pose(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.has_goal = True

        current_goal = (round(self.goal_x, 3), round(self.goal_y, 3))
        if current_goal != self.last_goal:
            self.last_goal = current_goal
            self.goal_start_time = time.time()
            self.reached_once = False

    def on_cmd_raw(self, msg):
        self.cmd_raw_linear = msg.linear.x

    def on_cmd_safe(self, msg):
        self.cmd_safe_linear = msg.linear.x
        self.cmd_safe_angular = msg.angular.z

    def on_safety_status(self, msg):
        self.safety_status = msg.data

    def on_dynamic_obstacles(self, msg):
        obstacles = []

        for marker in msg.markers:
            if marker.action == marker.DELETE:
                continue

            ox = marker.pose.position.x
            oy = marker.pose.position.y
            radius = max(marker.scale.x, marker.scale.y) * 0.5
            obstacles.append((ox, oy, radius))

        self.dynamic_obstacles = obstacles

    def select_local_path(self):
        now = self.now_seconds()

        teb_age = now - self.last_teb_local_plan_time
        trajectory_age = now - self.last_local_trajectory_time

        if self.teb_local_plan and teb_age <= self.local_path_timeout:
            return self.teb_local_plan, "/local_plan", teb_age

        if self.local_trajectory_path and trajectory_age <= self.local_path_timeout:
            return self.local_trajectory_path, "/local_trajectory", trajectory_age

        if self.teb_local_plan:
            return self.teb_local_plan, "/local_plan_stale", teb_age

        if self.local_trajectory_path:
            return self.local_trajectory_path, "/local_trajectory_stale", trajectory_age

        return [], "none", -1.0

    def path_length(self, points):
        if len(points) < 2:
            return 0.0

        total = 0.0
        for i in range(len(points) - 1):
            total += math.hypot(
                points[i + 1][0] - points[i][0],
                points[i + 1][1] - points[i][1],
            )

        return total

    def goal_distance(self):
        if not self.has_robot_pose or not self.has_goal:
            return -1.0

        return math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)

    def min_dynamic_obstacle_distance(self):
        if not self.has_robot_pose or not self.dynamic_obstacles:
            return -1.0

        nearest = None

        for ox, oy, radius in self.dynamic_obstacles:
            distance = math.hypot(ox - self.robot_x, oy - self.robot_y) - radius
            if nearest is None or distance < nearest:
                nearest = distance

        return nearest if nearest is not None else -1.0

    def navigation_status(self, goal_dist, local_path):
        if not self.has_robot_pose:
            return "WAITING_FOR_ROBOT_POSE"

        if not self.has_goal:
            return "WAITING_FOR_GOAL"

        if self.safety_status.startswith("STOP"):
            return "SAFETY_STOP"

        if goal_dist >= 0.0 and goal_dist < self.goal_tolerance:
            self.reached_once = True
            return "REACHED"

        if len(self.global_path) < 2:
            return "NO_GLOBAL_PATH"

        if len(local_path) < 2:
            return "NO_LOCAL_TRAJECTORY"

        return "NAVIGATING"

    def publish_float(self, publisher, value):
        msg = Float32()
        msg.data = float(value)
        publisher.publish(msg)

    def publish_string(self, publisher, text):
        msg = String()
        msg.data = text
        publisher.publish(msg)

    def publish_metrics(self):
        local_path, local_source, local_age = self.select_local_path()

        goal_dist = self.goal_distance()
        global_len = self.path_length(self.global_path)
        local_len = self.path_length(local_path)
        min_obs = self.min_dynamic_obstacle_distance()

        if abs(self.cmd_raw_linear) > 1e-4:
            safety_scale = self.cmd_safe_linear / self.cmd_raw_linear
        else:
            safety_scale = 0.0

        safety_scale = max(0.0, min(1.0, safety_scale))

        nav_status = self.navigation_status(goal_dist, local_path)
        elapsed = time.time() - self.goal_start_time

        self.publish_float(self.goal_distance_pub, goal_dist)
        self.publish_float(self.global_path_length_pub, global_len)
        self.publish_float(self.local_path_length_pub, local_len)
        self.publish_float(self.raw_linear_pub, self.cmd_raw_linear)
        self.publish_float(self.safe_linear_pub, self.cmd_safe_linear)
        self.publish_float(self.safe_angular_pub, self.cmd_safe_angular)
        self.publish_float(self.safety_scale_pub, safety_scale)
        self.publish_float(self.min_obstacle_pub, min_obs)
        self.publish_string(self.nav_status_pub, nav_status)
        self.publish_string(self.local_path_source_pub, local_source)

        summary = {
            "navigation_status": nav_status,
            "safety_status": self.safety_status,
            "goal_distance_m": round(goal_dist, 3),
            "global_path_length_m": round(global_len, 3),
            "local_trajectory_length_m": round(local_len, 3),
            "local_trajectory_source": local_source,
            "local_trajectory_age_s": round(local_age, 3),
            "cmd_raw_linear_mps": round(self.cmd_raw_linear, 3),
            "cmd_safe_linear_mps": round(self.cmd_safe_linear, 3),
            "cmd_safe_angular_radps": round(self.cmd_safe_angular, 3),
            "safety_scale": round(safety_scale, 3),
            "min_dynamic_obstacle_distance_m": round(min_obs, 3),
            "elapsed_since_goal_s": round(elapsed, 2),
        }

        self.publish_string(self.summary_pub, json.dumps(summary))


def main():
    rclpy.init()
    node = MetricsMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
