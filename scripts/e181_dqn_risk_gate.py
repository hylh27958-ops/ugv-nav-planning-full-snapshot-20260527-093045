#!/usr/bin/env python3
import json

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


def as_float(v, default=-1.0):
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


class E181DqnRiskGate(Node):
    def __init__(self):
        super().__init__("e181_dqn_risk_gate")

        self.declare_parameter("risk_gate_distance", 0.80)
        self.declare_parameter("safe_speed_scale", 0.85)
        self.declare_parameter("safe_lookahead_scale", 1.10)
        self.declare_parameter("safe_local_horizon_scale", 1.20)
        self.declare_parameter("safe_obstacle_caution", 1.35)

        self.risk_gate_distance = float(self.get_parameter("risk_gate_distance").value)
        self.safe_speed_scale = float(self.get_parameter("safe_speed_scale").value)
        self.safe_lookahead_scale = float(self.get_parameter("safe_lookahead_scale").value)
        self.safe_local_horizon_scale = float(self.get_parameter("safe_local_horizon_scale").value)
        self.safe_obstacle_caution = float(self.get_parameter("safe_obstacle_caution").value)

        self.latest_min_obs = -1.0

        self.pub = self.create_publisher(String, "/sac/adaptation", 10)
        self.create_subscription(String, "/sac/state", self.on_state, 10)
        self.create_subscription(String, "/sac/adaptation_dqn_raw", self.on_raw_adaptation, 10)

        self.get_logger().info(
            f"E18.1 DQN risk gate started. risk_gate_distance={self.risk_gate_distance:.2f}"
        )

    def on_state(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        self.latest_min_obs = as_float(
            data.get("min_dynamic_obstacle_distance_m", data.get("min_obstacle_m", -1.0)),
            -1.0,
        )

    def on_raw_adaptation(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        policy_source = str(data.get("policy_source", ""))
        rl_intent = as_bool(data.get("safety_trigger_rl_intent", False)) or ("safe" in policy_source)

        min_obs = self.latest_min_obs
        real_risk = 0.0 <= min_obs < self.risk_gate_distance
        active = bool(rl_intent and real_risk)

        data["schema_version"] = int(data.get("schema_version", 1))
        data["mode"] = "e181_gated_dqn_safety_trigger"
        data["risk_gate_distance"] = self.risk_gate_distance
        data["risk_gate_min_obstacle_m"] = round(min_obs, 4)
        data["safety_trigger_rl_intent"] = bool(rl_intent)
        data["safety_trigger_real_risk"] = bool(real_risk)
        data["safety_trigger_active"] = bool(active)

        if active:
            data["policy_source"] = "dqn_safe_mode_gated"
            data["safety_trigger_reason"] = "dqn_intent_and_real_risk"
            data["speed_scale"] = self.safe_speed_scale
            data["lookahead_scale"] = self.safe_lookahead_scale
            data["local_horizon_scale"] = self.safe_local_horizon_scale
            data["obstacle_caution"] = self.safe_obstacle_caution
        else:
            data["policy_source"] = "dqn_baseline_risk_gate" if rl_intent else "dqn_baseline"
            data["safety_trigger_reason"] = "risk_gate_blocked" if rl_intent else "dqn_baseline"
            data["speed_scale"] = 1.0
            data["lookahead_scale"] = 1.0
            data["local_horizon_scale"] = 1.0
            data["obstacle_caution"] = 1.0

        out = String()
        out.data = json.dumps(data, separators=(",", ":"))
        self.pub.publish(out)


def main():
    rclpy.init()
    node = E181DqnRiskGate()
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
