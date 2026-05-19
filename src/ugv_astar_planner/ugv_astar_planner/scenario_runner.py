import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, String


class ScenarioRunner(Node):
    def __init__(self):
        super().__init__("scenario_runner")

        self.declare_parameter(
            "goals",
            "12.0,1.0;1.0,8.0;12.8,8.0;0.8,0.8",
        )
        self.declare_parameter("loop", False)
        self.declare_parameter("timeout_sec", 70.0)
        self.declare_parameter("settle_sec", 2.0)
        self.declare_parameter("initial_delay_sec", 3.0)
        self.declare_parameter("publish_period_sec", 1.0)

        self.goals = self.parse_goals(self.get_parameter("goals").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.timeout_sec = float(self.get_parameter("timeout_sec").value)
        self.settle_sec = float(self.get_parameter("settle_sec").value)
        self.initial_delay_sec = float(self.get_parameter("initial_delay_sec").value)
        self.publish_period_sec = float(self.get_parameter("publish_period_sec").value)

        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

        self.create_subscription(String, "/metrics/navigation_status", self.on_nav_status, 10)
        self.create_subscription(Float32, "/metrics/goal_distance", self.on_goal_distance, 10)

        self.nav_status = "UNKNOWN"
        self.goal_distance = -1.0

        self.started_at = time.time()
        self.current_index = -1
        self.current_goal_start = None
        self.last_goal_publish = 0.0
        self.reached_at = None
        self.finished = False

        self.timer = self.create_timer(0.2, self.update)

        self.get_logger().info(f"Scenario runner loaded {len(self.goals)} goals.")

    def parse_goals(self, text):
        goals = []

        for item in text.split(";"):
            item = item.strip()
            if not item:
                continue

            parts = item.split(",")
            if len(parts) != 2:
                continue

            goals.append((float(parts[0]), float(parts[1])))

        return goals

    def on_nav_status(self, msg):
        self.nav_status = msg.data

    def on_goal_distance(self, msg):
        self.goal_distance = msg.data

    def publish_goal(self, goal):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = goal[0]
        msg.pose.position.y = goal[1]
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

    def start_next_goal(self):
        if not self.goals:
            self.get_logger().warn("No goals configured.")
            self.finished = True
            return

        self.current_index += 1

        if self.current_index >= len(self.goals):
            if self.loop:
                self.current_index = 0
            else:
                self.finished = True
                self.get_logger().info("Scenario finished.")
                return

        goal = self.goals[self.current_index]
        self.current_goal_start = time.time()
        self.last_goal_publish = 0.0
        self.reached_at = None

        self.get_logger().info(
            f"Starting goal {self.current_index + 1}/{len(self.goals)}: "
            f"x={goal[0]:.2f}, y={goal[1]:.2f}"
        )
        self.publish_goal(goal)

    def update(self):
        now = time.time()

        if self.finished:
            return

        if now - self.started_at < self.initial_delay_sec:
            return

        if self.current_index < 0:
            self.start_next_goal()
            return

        goal = self.goals[self.current_index]

        if now - self.last_goal_publish > self.publish_period_sec:
            self.publish_goal(goal)
            self.last_goal_publish = now

        if self.nav_status == "REACHED":
            if self.reached_at is None:
                self.reached_at = now
                self.get_logger().info(
                    f"Reached goal {self.current_index + 1}, "
                    f"distance={self.goal_distance:.2f} m"
                )

            if now - self.reached_at >= self.settle_sec:
                self.start_next_goal()

            return

        if self.current_goal_start is not None:
            elapsed = now - self.current_goal_start
            if elapsed > self.timeout_sec:
                self.get_logger().warn(
                    f"Goal {self.current_index + 1} timed out after {elapsed:.1f}s. "
                    "Moving to next goal."
                )
                self.start_next_goal()


def main():
    rclpy.init()
    node = ScenarioRunner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()