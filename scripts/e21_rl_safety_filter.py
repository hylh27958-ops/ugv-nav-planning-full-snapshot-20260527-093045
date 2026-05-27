#!/usr/bin/env python3
import json
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray


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


class E21RlSafetyFilter(Node):
    def __init__(self):
        super().__init__("e21_rl_safety_filter")

        self.declare_parameter("update_period", 0.05)
        self.declare_parameter("max_linear", 0.45)
        self.declare_parameter("max_angular", 1.2)
        self.declare_parameter("max_linear_accel", 0.70)
        self.declare_parameter("max_angular_accel", 2.0)
        self.declare_parameter("cmd_timeout", 1.5)
        self.declare_parameter("front_angle_deg", 45.0)

        self.declare_parameter("baseline_scale", 1.0)
        self.declare_parameter("rl_safe_scale", 0.55)
        self.declare_parameter("inactive_rl_scale", 1.0)
        self.declare_parameter("emergency_stop_distance", 0.0)

        self.update_period = float(self.get_parameter("update_period").value)
        self.max_linear = float(self.get_parameter("max_linear").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.max_linear_accel = float(self.get_parameter("max_linear_accel").value)
        self.max_angular_accel = float(self.get_parameter("max_angular_accel").value)
        self.cmd_timeout = float(self.get_parameter("cmd_timeout").value)
        self.front_angle = math.radians(float(self.get_parameter("front_angle_deg").value))

        self.baseline_scale = float(self.get_parameter("baseline_scale").value)
        self.rl_safe_scale = float(self.get_parameter("rl_safe_scale").value)
        self.inactive_rl_scale = float(self.get_parameter("inactive_rl_scale").value)
        self.emergency_stop_distance = float(self.get_parameter("emergency_stop_distance").value)

        self.raw_cmd = Twist()
        self.last_raw_time = self.get_clock().now()
        self.last_out = Twist()
        self.last_update_time = self.get_clock().now()

        self.has_pose = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.dynamic_obstacles = []

        self.rl_active = False
        self.rl_intent = False
        self.rl_policy_source = "no_rl_message"
        self.rl_reason = "no_rl_message"
        self.rl_speed_scale = 1.0
        self.rl_obstacle_caution = 1.0

        self.create_subscription(Twist, "/cmd_vel_raw", self.on_cmd_raw, 10)
        self.create_subscription(PoseStamped, "/robot_pose", self.on_robot_pose, 10)
        self.create_subscription(MarkerArray, "/dynamic_obstacles", self.on_dynamic_obstacles, 10)
        self.create_subscription(String, "/sac/adaptation", self.on_adaptation, 10)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.status_pub = self.create_publisher(String, "/safety_status", 10)

        self.timer = self.create_timer(self.update_period, self.update)
        self.get_logger().warn("E21 RL safety filter replacement active: /cmd_vel_raw -> /cmd_vel")

    def on_cmd_raw(self, msg):
        self.raw_cmd = msg
        self.last_raw_time = self.get_clock().now()

    def on_robot_pose(self, msg):
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        q = msg.pose.orientation
        self.yaw = self.yaw_from_quat(q.x, q.y, q.z, q.w)
        self.has_pose = True

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

    def on_adaptation(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            return

        self.rl_policy_source = str(data.get("policy_source", "unknown"))
        self.rl_reason = str(data.get("safety_trigger_reason", "unknown"))
        self.rl_active = as_bool(data.get("safety_trigger_active", False))
        self.rl_intent = as_bool(data.get("safety_trigger_rl_intent", False))
        self.rl_speed_scale = as_float(data.get("speed_scale", data.get("torch_speed_scale", 1.0)), 1.0)
        self.rl_obstacle_caution = as_float(data.get("obstacle_caution", data.get("torch_obstacle_caution", 1.0)), 1.0)

    def yaw_from_quat(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def clamp(self, value, lo, hi):
        return max(lo, min(hi, value))

    def approach(self, current, target, max_delta):
        if target > current + max_delta:
            return current + max_delta
        if target < current - max_delta:
            return current - max_delta
        return target

    def nearest_front_obstacle_distance(self):
        if not self.has_pose:
            return None

        nearest = None
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)

        for ox, oy, radius in self.dynamic_obstacles:
            dx = ox - self.x
            dy = oy - self.y

            x_r = cos_yaw * dx + sin_yaw * dy
            y_r = -sin_yaw * dx + cos_yaw * dy

            if x_r <= 0.0:
                continue

            angle = abs(math.atan2(y_r, x_r))
            if angle > self.front_angle:
                continue

            dist = max(0.0, math.hypot(x_r, y_r) - radius)
            if nearest is None or dist < nearest:
                nearest = dist

        return nearest

    def update(self):
        now = self.get_clock().now()
        dt = max(1e-3, (now - self.last_update_time).nanoseconds / 1e9)
        self.last_update_time = now

        age = (now - self.last_raw_time).nanoseconds / 1e9
        min_front_obs = self.nearest_front_obstacle_distance()

        target = Twist()
        reason = "baseline"
        emergency = False

        if age > self.cmd_timeout:
            reason = "cmd_timeout"
        else:
            if self.rl_active:
                velocity_scale = self.rl_safe_scale
                reason = "rl_safe_mode"
            elif self.rl_intent:
                velocity_scale = min(self.baseline_scale, self.inactive_rl_scale)
                reason = "rl_intent_inactive"
            else:
                velocity_scale = self.baseline_scale

            if (
                self.emergency_stop_distance > 0.0
                and min_front_obs is not None
                and min_front_obs < self.emergency_stop_distance
            ):
                velocity_scale = 0.0
                reason = "emergency_stop_guard"
                emergency = True

            target.linear.x = self.clamp(
                self.raw_cmd.linear.x * velocity_scale,
                -self.max_linear,
                self.max_linear,
            )
            target.angular.z = self.clamp(
                self.raw_cmd.angular.z,
                -self.max_angular,
                self.max_angular,
            )

        out = Twist()
        out.linear.x = self.approach(
            self.last_out.linear.x,
            target.linear.x,
            self.max_linear_accel * dt,
        )
        out.angular.z = self.approach(
            self.last_out.angular.z,
            target.angular.z,
            self.max_angular_accel * dt,
        )

        self.last_out = out
        self.cmd_pub.publish(out)

        status = {
            "mode": "e21_rl_safety_filter_replacement",
            "reason": reason,
            "rl_active": bool(self.rl_active),
            "rl_intent": bool(self.rl_intent),
            "rl_policy_source": self.rl_policy_source,
            "rl_reason": self.rl_reason,
            "rl_speed_scale": round(self.rl_speed_scale, 4),
            "rl_obstacle_caution": round(self.rl_obstacle_caution, 4),
            "min_front_obstacle_m": None if min_front_obs is None else round(min_front_obs, 4),
            "emergency": bool(emergency),
        }

        msg = String()
        msg.data = json.dumps(status, separators=(",", ":"))
        self.status_pub.publish(msg)


def main():
    rclpy.init()
    node = E21RlSafetyFilter()
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
