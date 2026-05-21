#!/usr/bin/env python3
import json
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String


class SacSafetyGate(Node):
    def __init__(self):
        super().__init__("sac_safety_gate")

        self.declare_parameter("input_topic", "/sac/adaptation_raw")
        self.declare_parameter("output_topic", "/sac/adaptation")
        self.declare_parameter("status_topic", "/sac/safety_gate_status")
        self.declare_parameter("obstacle_topic", "/metrics/min_dynamic_obstacle_distance")

        self.declare_parameter("caution_distance", 0.50)
        self.declare_parameter("danger_distance", 0.20)
        self.declare_parameter("emergency_distance", 0.10)

        self.declare_parameter("caution_speed_cap", 1.00)
        self.declare_parameter("danger_speed_cap", 0.70)
        self.declare_parameter("emergency_speed_cap", 0.45)

        self.declare_parameter("caution_obstacle_caution", 1.45)
        self.declare_parameter("danger_obstacle_caution", 1.75)
        self.declare_parameter("emergency_obstacle_caution", 1.90)

        self.declare_parameter("caution_horizon_min", 1.15)
        self.declare_parameter("danger_horizon_min", 1.30)
        self.declare_parameter("emergency_horizon_min", 1.45)

        self.declare_parameter("stale_timeout_s", 1.0)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.status_topic = self.get_parameter("status_topic").value
        self.obstacle_topic = self.get_parameter("obstacle_topic").value

        self.min_obstacle = -1.0
        self.last_obstacle_time = 0.0
        self.last_status = {
            "active": False,
            "reason": "waiting_for_obstacle_metric",
        }

        self.create_subscription(Float32, self.obstacle_topic, self.on_obstacle, 10)
        self.create_subscription(String, self.input_topic, self.on_adaptation, 10)

        self.adaptation_pub = self.create_publisher(String, self.output_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.create_timer(1.0, self.publish_status)

        self.get_logger().info(
            f"SAC safety gate started: {self.input_topic} -> {self.output_topic}"
        )

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_obstacle(self, msg):
        self.min_obstacle = float(msg.data)
        self.last_obstacle_time = self.now_s()

    def clamp(self, value, lo, hi):
        return max(lo, min(hi, value))

    def read_action_value(self, data, name, default):
        action = data.get("action")
        if isinstance(action, dict) and name in action:
            try:
                return float(action[name])
            except (TypeError, ValueError):
                return default

        if name in data:
            try:
                return float(data[name])
            except (TypeError, ValueError):
                return default

        return default

    def write_action_value(self, data, name, value):
        value = float(value)

        action = data.get("action")
        if isinstance(action, dict):
            action[name] = value

        data[name] = value

    def interpolate(self, x, x0, x1, y0, y1):
        if x1 <= x0:
            return y1
        ratio = self.clamp((x - x0) / (x1 - x0), 0.0, 1.0)
        return y0 + ratio * (y1 - y0)

    def gate_values(self, obstacle_distance):
        caution_d = float(self.get_parameter("caution_distance").value)
        danger_d = float(self.get_parameter("danger_distance").value)
        emergency_d = float(self.get_parameter("emergency_distance").value)

        caution_speed = float(self.get_parameter("caution_speed_cap").value)
        danger_speed = float(self.get_parameter("danger_speed_cap").value)
        emergency_speed = float(self.get_parameter("emergency_speed_cap").value)

        caution_obs = float(self.get_parameter("caution_obstacle_caution").value)
        danger_obs = float(self.get_parameter("danger_obstacle_caution").value)
        emergency_obs = float(self.get_parameter("emergency_obstacle_caution").value)

        caution_horizon = float(self.get_parameter("caution_horizon_min").value)
        danger_horizon = float(self.get_parameter("danger_horizon_min").value)
        emergency_horizon = float(self.get_parameter("emergency_horizon_min").value)

        if obstacle_distance < 0.0 or obstacle_distance >= caution_d:
            return {
                "active": False,
                "speed_cap": 1.0,
                "obstacle_caution_min": 1.0,
                "local_horizon_min": 1.0,
                "lookahead_min": 1.0,
                "level": "clear",
            }

        if obstacle_distance <= emergency_d:
            return {
                "active": True,
                "speed_cap": emergency_speed,
                "obstacle_caution_min": emergency_obs,
                "local_horizon_min": emergency_horizon,
                "lookahead_min": 1.08,
                "level": "emergency",
            }

        if obstacle_distance <= danger_d:
            speed_cap = self.interpolate(
                obstacle_distance,
                emergency_d,
                danger_d,
                emergency_speed,
                danger_speed,
            )
            obs_min = self.interpolate(
                obstacle_distance,
                emergency_d,
                danger_d,
                emergency_obs,
                danger_obs,
            )
            horizon_min = self.interpolate(
                obstacle_distance,
                emergency_d,
                danger_d,
                emergency_horizon,
                danger_horizon,
            )
            return {
                "active": True,
                "speed_cap": speed_cap,
                "obstacle_caution_min": obs_min,
                "local_horizon_min": horizon_min,
                "lookahead_min": 1.05,
                "level": "danger",
            }

        speed_cap = self.interpolate(
            obstacle_distance,
            danger_d,
            caution_d,
            danger_speed,
            caution_speed,
        )
        obs_min = self.interpolate(
            obstacle_distance,
            danger_d,
            caution_d,
            danger_obs,
            caution_obs,
        )
        horizon_min = self.interpolate(
            obstacle_distance,
            danger_d,
            caution_d,
            danger_horizon,
            caution_horizon,
        )

        return {
            "active": True,
            "speed_cap": speed_cap,
            "obstacle_caution_min": obs_min,
            "local_horizon_min": horizon_min,
            "lookahead_min": 1.0,
            "level": "caution",
        }

    def on_adaptation(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.adaptation_pub.publish(msg)
            self.last_status = {
                "active": False,
                "reason": "invalid_json_passthrough",
            }
            return

        stale_timeout = float(self.get_parameter("stale_timeout_s").value)
        age = self.now_s() - self.last_obstacle_time

        if self.last_obstacle_time <= 0.0 or age > stale_timeout:
            self.adaptation_pub.publish(msg)
            self.last_status = {
                "active": False,
                "reason": "obstacle_metric_stale",
                "age_s": round(age, 3),
            }
            return

        gate = self.gate_values(self.min_obstacle)

        input_speed = self.read_action_value(data, "speed_scale", 1.0)
        input_lookahead = self.read_action_value(data, "lookahead_scale", 1.0)
        input_horizon = self.read_action_value(data, "local_horizon_scale", 1.0)
        input_caution = self.read_action_value(data, "obstacle_caution", 1.0)

        output_speed = min(input_speed, gate["speed_cap"])
        output_lookahead = max(input_lookahead, gate["lookahead_min"])
        output_horizon = max(input_horizon, gate["local_horizon_min"])
        output_caution = max(input_caution, gate["obstacle_caution_min"])

        output_speed = self.clamp(output_speed, 0.20, 1.05)
        output_lookahead = self.clamp(output_lookahead, 0.70, 1.50)
        output_horizon = self.clamp(output_horizon, 0.80, 1.70)
        output_caution = self.clamp(output_caution, 1.00, 2.00)

        self.write_action_value(data, "speed_scale", output_speed)
        self.write_action_value(data, "lookahead_scale", output_lookahead)
        self.write_action_value(data, "local_horizon_scale", output_horizon)
        self.write_action_value(data, "obstacle_caution", output_caution)

        data["safety_gate"] = {
            "enabled": True,
            "active": gate["active"],
            "level": gate["level"],
            "min_dynamic_obstacle_distance_m": round(self.min_obstacle, 3),
            "input_speed_scale": round(input_speed, 3),
            "output_speed_scale": round(output_speed, 3),
            "speed_cap": round(gate["speed_cap"], 3),
            "input_obstacle_caution": round(input_caution, 3),
            "output_obstacle_caution": round(output_caution, 3),
        }

        out = String()
        out.data = json.dumps(data, separators=(",", ":"))
        self.adaptation_pub.publish(out)

        self.last_status = data["safety_gate"]

    def publish_status(self):
        msg = String()
        msg.data = json.dumps(self.last_status, separators=(",", ":"))
        self.status_pub.publish(msg)


def main():
    rclpy.init()
    node = SacSafetyGate()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
