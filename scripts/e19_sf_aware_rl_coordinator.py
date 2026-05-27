#!/usr/bin/env python3
import json

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


def as_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes", "on")
    return False


class E19SfAwareRlCoordinator(Node):
    def __init__(self):
        super().__init__("e19_sf_aware_rl_coordinator")

        self.declare_parameter("warning_min_obstacle", 0.20)
        self.declare_parameter("warning_max_obstacle", 0.90)
        self.declare_parameter("hard_filter_scale", 0.35)
        self.declare_parameter("clear_filter_scale", 0.95)

        self.declare_parameter("rl_speed_intent_threshold", 0.88)
        self.declare_parameter("rl_caution_intent_threshold", 1.25)

        self.declare_parameter("safe_speed_scale", 0.90)
        self.declare_parameter("safe_lookahead_scale", 1.08)
        self.declare_parameter("safe_local_horizon_scale", 1.15)
        self.declare_parameter("safe_obstacle_caution", 1.20)

        self.warning_min_obstacle = float(self.get_parameter("warning_min_obstacle").value)
        self.warning_max_obstacle = float(self.get_parameter("warning_max_obstacle").value)
        self.hard_filter_scale = float(self.get_parameter("hard_filter_scale").value)
        self.clear_filter_scale = float(self.get_parameter("clear_filter_scale").value)

        self.rl_speed_intent_threshold = float(self.get_parameter("rl_speed_intent_threshold").value)
        self.rl_caution_intent_threshold = float(self.get_parameter("rl_caution_intent_threshold").value)

        self.safe_speed_scale = float(self.get_parameter("safe_speed_scale").value)
        self.safe_lookahead_scale = float(self.get_parameter("safe_lookahead_scale").value)
        self.safe_local_horizon_scale = float(self.get_parameter("safe_local_horizon_scale").value)
        self.safe_obstacle_caution = float(self.get_parameter("safe_obstacle_caution").value)

        self.latest_state = {}

        self.pub = self.create_publisher(String, "/sac/adaptation", 10)
        self.create_subscription(String, "/sac/state", self.on_state, 10)
        self.create_subscription(String, "/sac/adaptation_rl_raw", self.on_raw_adaptation, 10)

        self.get_logger().info("E19 safety-filter-aware RL coordinator started")

    def on_state(self, msg):
        try:
            self.latest_state = json.loads(msg.data)
        except Exception:
            self.latest_state = {}

    def on_raw_adaptation(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        min_obs = as_float(
            self.latest_state.get("min_dynamic_obstacle_distance_m",
            self.latest_state.get("min_obstacle_m", -1.0)),
            -1.0,
        )
        safety_scale = as_float(self.latest_state.get("safety_scale", 1.0), 1.0)
        safe_speed = as_float(self.latest_state.get("safe_linear_mps", 0.0), 0.0)

        torch_speed = as_float(data.get("torch_speed_scale", data.get("speed_scale", 1.0)), 1.0)
        torch_caution = as_float(data.get("torch_obstacle_caution", data.get("obstacle_caution", 1.0)), 1.0)

        rl_intent = (
            as_bool(data.get("safety_trigger_rl_intent", False))
            or torch_speed < self.rl_speed_intent_threshold
            or torch_caution > self.rl_caution_intent_threshold
        )

        valid_obs = min_obs >= 0.0
        obstacle_warning = valid_obs and self.warning_min_obstacle <= min_obs < self.warning_max_obstacle
        hard_filter = safety_scale < self.hard_filter_scale or safe_speed < 0.08
        clear_filter = safety_scale >= self.clear_filter_scale
        warning_filter = (not hard_filter) and (not clear_filter)

        active = bool(rl_intent and obstacle_warning and warning_filter)

        if hard_filter:
            zone = "hard_filter"
        elif warning_filter:
            zone = "warning_filter"
        elif clear_filter:
            zone = "clear_filter"
        else:
            zone = "unknown"

        data["schema_version"] = int(data.get("schema_version", 1))
        data["mode"] = "e19_sf_aware_rl_coordination"
        data["policy_type"] = "sf_aware_rl_safety_trigger"

        data["sf_coordination_zone"] = zone
        data["sf_coordination_min_obstacle_m"] = round(min_obs, 4)
        data["sf_coordination_safety_scale"] = round(safety_scale, 4)
        data["sf_coordination_safe_speed_mps"] = round(safe_speed, 4)

        data["safety_trigger_rl_intent"] = bool(rl_intent)
        data["safety_trigger_real_risk"] = bool(obstacle_warning)
        data["safety_trigger_active"] = bool(active)

        data["torch_speed_scale"] = round(torch_speed, 4)
        data["torch_obstacle_caution"] = round(torch_caution, 4)

        if active:
            data["policy_source"] = "sf_aware_rl_safe_mode"
            data["safety_trigger_reason"] = "rl_intent_in_sf_warning_band"
            data["speed_scale"] = self.safe_speed_scale
            data["lookahead_scale"] = self.safe_lookahead_scale
            data["local_horizon_scale"] = self.safe_local_horizon_scale
            data["obstacle_caution"] = self.safe_obstacle_caution
        else:
            data["policy_source"] = "sf_aware_baseline"
            data["safety_trigger_reason"] = f"blocked_{zone}"
            data["speed_scale"] = 1.0
            data["lookahead_scale"] = 1.0
            data["local_horizon_scale"] = 1.0
            data["obstacle_caution"] = 1.0

        out = String()
        out.data = json.dumps(data, separators=(",", ":"))
        self.pub.publish(out)


def main():
    rclpy.init()
    node = E19SfAwareRlCoordinator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
