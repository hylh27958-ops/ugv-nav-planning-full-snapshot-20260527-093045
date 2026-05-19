import json

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32, String


class SacAdapter(Node):
    def __init__(self):
        super().__init__("sac_adapter")

        self.declare_parameter("update_period", 0.2)
        self.declare_parameter("smoothing_alpha", 0.25)
        self.declare_parameter("obstacle_caution_distance", 0.8)
        self.declare_parameter("goal_slow_distance", 1.0)

        self.update_period = float(self.get_parameter("update_period").value)
        self.smoothing_alpha = float(self.get_parameter("smoothing_alpha").value)
        self.obstacle_caution_distance = float(
            self.get_parameter("obstacle_caution_distance").value
        )
        self.goal_slow_distance = float(self.get_parameter("goal_slow_distance").value)

        self.create_subscription(
            Float32,
            "/metrics/min_dynamic_obstacle_distance",
            self.on_min_obstacle_distance,
            10,
        )
        self.create_subscription(
            Float32,
            "/metrics/goal_distance",
            self.on_goal_distance,
            10,
        )
        self.create_subscription(
            String,
            "/metrics/navigation_status",
            self.on_navigation_status,
            10,
        )
        self.create_subscription(
            String,
            "/safety_status",
            self.on_safety_status,
            10,
        )

        self.pub = self.create_publisher(String, "/sac/adaptation", 10)

        self.min_obstacle_distance = -1.0
        self.goal_distance = -1.0
        self.navigation_status = "UNKNOWN"
        self.safety_status = "UNKNOWN"

        self.current = {
            "max_linear_scale": 1.0,
            "lookahead_scale": 1.0,
            "horizon_scale": 1.0,
            "obstacle_influence_scale": 1.0,
            "obstacle_shift_scale": 1.0,
            "stop_distance_scale": 1.0,
            "slow_distance_scale": 1.0,
        }

        self.timer = self.create_timer(self.update_period, self.update)
        self.get_logger().info("SAC adapter interface started.")

    def on_min_obstacle_distance(self, msg):
        self.min_obstacle_distance = msg.data

    def on_goal_distance(self, msg):
        self.goal_distance = msg.data

    def on_navigation_status(self, msg):
        self.navigation_status = msg.data

    def on_safety_status(self, msg):
        self.safety_status = msg.data

    def blend(self, old, new):
        alpha = max(0.0, min(1.0, self.smoothing_alpha))
        return old * (1.0 - alpha) + new * alpha

    def build_target_action(self):
        if self.safety_status.startswith("STOP") or self.navigation_status == "SAFETY_STOP":
            return {
                "mode": "STOP_RECOVERY",
                "reason": self.safety_status,
                "max_linear_scale": 0.35,
                "lookahead_scale": 0.80,
                "horizon_scale": 0.80,
                "obstacle_influence_scale": 1.45,
                "obstacle_shift_scale": 1.50,
                "stop_distance_scale": 1.25,
                "slow_distance_scale": 1.30,
            }

        if (
            self.min_obstacle_distance >= 0.0
            and self.min_obstacle_distance < self.obstacle_caution_distance
        ):
            return {
                "mode": "CAUTIOUS",
                "reason": f"obstacle {self.min_obstacle_distance:.2f} m",
                "max_linear_scale": 0.50,
                "lookahead_scale": 0.85,
                "horizon_scale": 0.85,
                "obstacle_influence_scale": 1.35,
                "obstacle_shift_scale": 1.40,
                "stop_distance_scale": 1.15,
                "slow_distance_scale": 1.25,
            }

        if self.goal_distance >= 0.0 and self.goal_distance < self.goal_slow_distance:
            return {
                "mode": "GOAL_APPROACH",
                "reason": f"goal {self.goal_distance:.2f} m",
                "max_linear_scale": 0.45,
                "lookahead_scale": 0.75,
                "horizon_scale": 0.75,
                "obstacle_influence_scale": 1.05,
                "obstacle_shift_scale": 1.00,
                "stop_distance_scale": 1.00,
                "slow_distance_scale": 1.00,
            }

        return {
            "mode": "NORMAL",
            "reason": "clear",
            "max_linear_scale": 1.0,
            "lookahead_scale": 1.0,
            "horizon_scale": 1.0,
            "obstacle_influence_scale": 1.0,
            "obstacle_shift_scale": 1.0,
            "stop_distance_scale": 1.0,
            "slow_distance_scale": 1.0,
        }

    def update(self):
        target = self.build_target_action()

        for key in self.current:
            self.current[key] = self.blend(self.current[key], float(target[key]))

        output = {
            "mode": target["mode"],
            "reason": target["reason"],
            "max_linear_scale": round(self.current["max_linear_scale"], 3),
            "lookahead_scale": round(self.current["lookahead_scale"], 3),
            "horizon_scale": round(self.current["horizon_scale"], 3),
            "obstacle_influence_scale": round(
                self.current["obstacle_influence_scale"], 3
            ),
            "obstacle_shift_scale": round(self.current["obstacle_shift_scale"], 3),
            "stop_distance_scale": round(self.current["stop_distance_scale"], 3),
            "slow_distance_scale": round(self.current["slow_distance_scale"], 3),
            "min_obstacle_distance": round(self.min_obstacle_distance, 3),
            "goal_distance": round(self.goal_distance, 3),
            "navigation_status": self.navigation_status,
            "safety_status": self.safety_status,
        }

        msg = String()
        msg.data = json.dumps(output)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = SacAdapter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()