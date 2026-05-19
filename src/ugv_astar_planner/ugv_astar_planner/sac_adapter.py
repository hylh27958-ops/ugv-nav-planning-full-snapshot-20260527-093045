import json
import math
import time

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

        self.declare_parameter("min_speed_scale", 0.40)
        self.declare_parameter("max_speed_scale", 1.00)
        self.declare_parameter("min_lookahead_scale", 0.60)
        self.declare_parameter("max_lookahead_scale", 1.50)
        self.declare_parameter("min_obstacle_caution", 0.70)
        self.declare_parameter("max_obstacle_caution", 1.80)
        self.declare_parameter("min_local_horizon_scale", 0.70)
        self.declare_parameter("max_local_horizon_scale", 1.30)

        self.update_period = float(self.get_parameter("update_period").value)
        self.smoothing_alpha = float(self.get_parameter("smoothing_alpha").value)
        self.obstacle_caution_distance = float(
            self.get_parameter("obstacle_caution_distance").value
        )
        self.goal_slow_distance = float(self.get_parameter("goal_slow_distance").value)

        self.min_speed_scale = float(self.get_parameter("min_speed_scale").value)
        self.max_speed_scale = float(self.get_parameter("max_speed_scale").value)
        self.min_lookahead_scale = float(self.get_parameter("min_lookahead_scale").value)
        self.max_lookahead_scale = float(self.get_parameter("max_lookahead_scale").value)
        self.min_obstacle_caution = float(self.get_parameter("min_obstacle_caution").value)
        self.max_obstacle_caution = float(self.get_parameter("max_obstacle_caution").value)
        self.min_local_horizon_scale = float(
            self.get_parameter("min_local_horizon_scale").value
        )
        self.max_local_horizon_scale = float(
            self.get_parameter("max_local_horizon_scale").value
        )

        self.create_subscription(String, "/metrics/summary", self.on_metrics_summary, 10)

        self.adaptation_pub = self.create_publisher(String, "/sac/adaptation", 10)
        self.state_pub = self.create_publisher(String, "/sac/state", 10)
        self.reward_pub = self.create_publisher(Float32, "/sac/reward", 10)

        self.metrics = {}
        self.last_metrics_time = 0.0

        self.last_goal_distance = None
        self.last_reward_time = None

        self.speed_scale = 1.0
        self.lookahead_scale = 1.0
        self.obstacle_caution = 1.0
        self.local_horizon_scale = 1.0

        self.timer = self.create_timer(self.update_period, self.on_timer)

        self.get_logger().info("SAC adapter B1 RL parameter interface started.")

    def on_metrics_summary(self, msg):
        try:
            self.metrics = json.loads(msg.data)
            self.last_metrics_time = time.time()
        except json.JSONDecodeError:
            self.get_logger().warn("Failed to parse /metrics/summary JSON.")

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def smooth(self, old_value, target_value):
        alpha = self.clamp(self.smoothing_alpha, 0.0, 1.0)
        return (1.0 - alpha) * old_value + alpha * target_value

    def get_metric(self, key, default):
        value = self.metrics.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def normalized_obstacle_risk(self, min_obstacle_distance):
        if min_obstacle_distance < 0.0:
            return 0.0

        if self.obstacle_caution_distance <= 1e-6:
            return 0.0

        risk = (self.obstacle_caution_distance - min_obstacle_distance) / self.obstacle_caution_distance
        return self.clamp(risk, 0.0, 1.0)

    def normalized_goal_risk(self, goal_distance):
        if goal_distance < 0.0:
            return 0.0

        if self.goal_slow_distance <= 1e-6:
            return 0.0

        risk = (self.goal_slow_distance - goal_distance) / self.goal_slow_distance
        return self.clamp(risk, 0.0, 1.0)

    def build_state(self):
        now = time.time()
        metrics_age = now - self.last_metrics_time if self.last_metrics_time > 0.0 else -1.0

        state = {
            "schema_version": 1,
            "stamp": now,
            "metrics_age_s": round(metrics_age, 3),
            "navigation_status": self.metrics.get("navigation_status", "UNKNOWN"),
            "safety_status": self.metrics.get("safety_status", "UNKNOWN"),
            "goal_distance_m": self.get_metric("goal_distance_m", -1.0),
            "global_path_length_m": self.get_metric("global_path_length_m", -1.0),
            "local_trajectory_length_m": self.get_metric("local_trajectory_length_m", -1.0),
            "cmd_raw_linear_mps": self.get_metric("cmd_raw_linear_mps", 0.0),
            "cmd_safe_linear_mps": self.get_metric("cmd_safe_linear_mps", 0.0),
            "cmd_safe_angular_radps": self.get_metric("cmd_safe_angular_radps", 0.0),
            "safety_scale": self.get_metric("safety_scale", 1.0),
            "min_dynamic_obstacle_distance_m": self.get_metric(
                "min_dynamic_obstacle_distance_m", -1.0
            ),
            "elapsed_since_goal_s": self.get_metric("elapsed_since_goal_s", 0.0),
        }

        min_obs = state["min_dynamic_obstacle_distance_m"]
        goal_dist = state["goal_distance_m"]
        safety_scale = self.clamp(state["safety_scale"], 0.0, 1.0)

        state["obstacle_risk"] = round(self.normalized_obstacle_risk(min_obs), 3)
        state["goal_slow_risk"] = round(self.normalized_goal_risk(goal_dist), 3)
        state["safety_intervention"] = round(1.0 - safety_scale, 3)

        return state

    def compute_rule_policy_action(self, state):
        obstacle_risk = float(state["obstacle_risk"])
        goal_risk = float(state["goal_slow_risk"])
        safety_intervention = float(state["safety_intervention"])

        target_speed_scale = 1.0
        target_speed_scale -= 0.45 * obstacle_risk
        target_speed_scale -= 0.25 * goal_risk
        target_speed_scale -= 0.20 * safety_intervention
        target_speed_scale = self.clamp(
            target_speed_scale, self.min_speed_scale, self.max_speed_scale
        )

        target_lookahead_scale = 1.15
        target_lookahead_scale -= 0.35 * obstacle_risk
        target_lookahead_scale -= 0.25 * goal_risk
        target_lookahead_scale = self.clamp(
            target_lookahead_scale,
            self.min_lookahead_scale,
            self.max_lookahead_scale,
        )

        target_obstacle_caution = 1.0
        target_obstacle_caution += 0.65 * obstacle_risk
        target_obstacle_caution += 0.25 * safety_intervention
        target_obstacle_caution = self.clamp(
            target_obstacle_caution,
            self.min_obstacle_caution,
            self.max_obstacle_caution,
        )

        target_local_horizon_scale = 1.10
        target_local_horizon_scale -= 0.30 * obstacle_risk
        target_local_horizon_scale -= 0.15 * goal_risk
        target_local_horizon_scale = self.clamp(
            target_local_horizon_scale,
            self.min_local_horizon_scale,
            self.max_local_horizon_scale,
        )

        self.speed_scale = self.smooth(self.speed_scale, target_speed_scale)
        self.lookahead_scale = self.smooth(self.lookahead_scale, target_lookahead_scale)
        self.obstacle_caution = self.smooth(self.obstacle_caution, target_obstacle_caution)
        self.local_horizon_scale = self.smooth(
            self.local_horizon_scale, target_local_horizon_scale
        )

        return {
            "speed_scale": round(self.speed_scale, 3),
            "lookahead_scale": round(self.lookahead_scale, 3),
            "obstacle_caution": round(self.obstacle_caution, 3),
            "local_horizon_scale": round(self.local_horizon_scale, 3),
        }

    def compute_reward_hint(self, state):
        now = time.time()
        goal_distance = float(state["goal_distance_m"])
        min_obs = float(state["min_dynamic_obstacle_distance_m"])
        safety_intervention = float(state["safety_intervention"])
        safe_speed = float(state["cmd_safe_linear_mps"])

        progress_reward = 0.0
        if self.last_goal_distance is not None and goal_distance >= 0.0:
            progress_reward = self.last_goal_distance - goal_distance

        obstacle_penalty = 0.0
        if min_obs >= 0.0 and min_obs < self.obstacle_caution_distance:
            obstacle_penalty = (
                self.obstacle_caution_distance - min_obs
            ) / self.obstacle_caution_distance

        speed_reward = 0.15 * self.clamp(safe_speed / 0.45, 0.0, 1.0)
        safety_penalty = 0.30 * safety_intervention
        time_penalty = 0.01

        reward = progress_reward + speed_reward - obstacle_penalty - safety_penalty - time_penalty

        if state["navigation_status"] == "REACHED":
            reward += 5.0

        if state["navigation_status"] == "SAFETY_STOP":
            reward -= 1.0

        self.last_goal_distance = goal_distance if goal_distance >= 0.0 else self.last_goal_distance
        self.last_reward_time = now

        return {
            "reward": round(reward, 4),
            "progress_reward": round(progress_reward, 4),
            "speed_reward": round(speed_reward, 4),
            "obstacle_penalty": round(obstacle_penalty, 4),
            "safety_penalty": round(safety_penalty, 4),
            "time_penalty": round(time_penalty, 4),
        }

    def publish_string(self, publisher, payload):
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        publisher.publish(msg)

    def publish_reward(self, reward):
        msg = Float32()
        msg.data = float(reward)
        self.reward_pub.publish(msg)

    def on_timer(self):
        state = self.build_state()
        action = self.compute_rule_policy_action(state)
        reward_info = self.compute_reward_hint(state)

        adaptation = {
            "schema_version": 1,
            "mode": "rule_policy",
            "policy_type": "rl_parameter_interface",
            "stamp": state["stamp"],
            "state": state,
            "action": action,
            "reward_hint": reward_info,
            "speed_scale": action["speed_scale"],
            "lookahead_scale": action["lookahead_scale"],
            "obstacle_caution": action["obstacle_caution"],
            "local_horizon_scale": action["local_horizon_scale"],
        }

        self.publish_string(self.state_pub, state)
        self.publish_string(self.adaptation_pub, adaptation)
        self.publish_reward(reward_info["reward"])


def main():
    rclpy.init()
    node = SacAdapter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
