import json
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float32, String
from visualization_msgs.msg import MarkerArray


class OnlineSacSafetyPolicy(Node):
    def __init__(self):
        super().__init__("online_sac_safety_policy")

        self.declare_parameter("update_period", 0.05)
        self.declare_parameter("max_linear", 0.45)
        self.declare_parameter("max_angular", 1.2)
        self.declare_parameter("cmd_timeout", 0.5)
        self.declare_parameter("action_mode", "rule_bootstrap")
        self.declare_parameter("training_enabled", False)
        self.declare_parameter("publish_debug", True)

        self.declare_parameter("caution_distance", 0.85)
        self.declare_parameter("warning_distance", 0.45)
        self.declare_parameter("emergency_distance", 0.20)
        self.declare_parameter("min_linear_scale", 0.0)
        self.declare_parameter("max_linear_scale", 1.0)
        self.declare_parameter("min_angular_scale", 0.5)
        self.declare_parameter("max_angular_scale", 1.2)
        self.declare_parameter("max_angular_residual", 0.25)

        self.declare_parameter("map_min_x", 0.0)
        self.declare_parameter("map_max_x", 14.0)
        self.declare_parameter("map_min_y", 0.0)
        self.declare_parameter("map_max_y", 9.0)
        self.declare_parameter("boundary_margin", 0.25)
        self.declare_parameter("boundary_caution_margin", 0.60)

        self.update_period = float(self.get_parameter("update_period").value)
        self.max_linear = float(self.get_parameter("max_linear").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.cmd_timeout = float(self.get_parameter("cmd_timeout").value)
        self.action_mode = str(self.get_parameter("action_mode").value)
        self.training_enabled = bool(self.get_parameter("training_enabled").value)
        self.publish_debug = bool(self.get_parameter("publish_debug").value)

        self.caution_distance = float(self.get_parameter("caution_distance").value)
        self.warning_distance = float(self.get_parameter("warning_distance").value)
        self.emergency_distance = float(self.get_parameter("emergency_distance").value)
        self.min_linear_scale = float(self.get_parameter("min_linear_scale").value)
        self.max_linear_scale = float(self.get_parameter("max_linear_scale").value)
        self.min_angular_scale = float(self.get_parameter("min_angular_scale").value)
        self.max_angular_scale = float(self.get_parameter("max_angular_scale").value)
        self.max_angular_residual = float(self.get_parameter("max_angular_residual").value)

        self.map_min_x = float(self.get_parameter("map_min_x").value)
        self.map_max_x = float(self.get_parameter("map_max_x").value)
        self.map_min_y = float(self.get_parameter("map_min_y").value)
        self.map_max_y = float(self.get_parameter("map_max_y").value)
        self.boundary_margin = float(self.get_parameter("boundary_margin").value)
        self.boundary_caution_margin = float(self.get_parameter("boundary_caution_margin").value)

        self.raw_cmd = Twist()
        self.last_raw_time = self.get_clock().now()

        self.has_pose = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.has_goal = False
        self.goal_x = 0.0
        self.goal_y = 0.0

        self.dynamic_obstacles = []
        self.metric_goal_distance = -1.0
        self.metric_min_obstacle = -1.0

        self.prev_goal_distance = None
        self.progress_mps = 0.0
        self.prev_time_s = self.now_s()
        self.prev_action = {
            "linear_scale": 1.0,
            "angular_scale": 1.0,
            "angular_residual": 0.0,
        }

        self.create_subscription(Twist, "/cmd_vel_raw", self.on_cmd_raw, 10)
        self.create_subscription(PoseStamped, "/robot_pose", self.on_robot_pose, 10)
        self.create_subscription(PoseStamped, "/astar_goal", self.on_goal_pose, 10)
        self.create_subscription(MarkerArray, "/dynamic_obstacles", self.on_dynamic_obstacles, 10)
        self.create_subscription(Float32, "/metrics/goal_distance", self.on_metric_goal_distance, 10)
        self.create_subscription(Float32, "/metrics/min_dynamic_obstacle_distance", self.on_metric_min_obstacle, 10)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel_sac", 10)
        self.state_pub = self.create_publisher(String, "/rl/safety_state", 10)
        self.action_pub = self.create_publisher(String, "/rl/safety_action", 10)
        self.reward_pub = self.create_publisher(Float32, "/rl/safety_reward", 10)
        self.status_pub = self.create_publisher(String, "/rl/training_status", 10)

        self.timer = self.create_timer(self.update_period, self.update)
        self.get_logger().info("Online SAC safety policy bootstrap started: /cmd_vel_raw -> /cmd_vel_sac")

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def clamp(self, value, lo, hi):
        return max(lo, min(hi, value))

    def yaw_from_quat(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def on_cmd_raw(self, msg):
        self.raw_cmd = msg
        self.last_raw_time = self.get_clock().now()

    def on_robot_pose(self, msg):
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.yaw = self.yaw_from_quat(msg.pose.orientation)
        self.has_pose = True

    def on_goal_pose(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.has_goal = True

    def on_dynamic_obstacles(self, msg):
        obstacles = []
        for marker in msg.markers:
            if marker.action == marker.DELETE:
                continue
            radius = max(marker.scale.x, marker.scale.y) * 0.5
            obstacles.append((marker.pose.position.x, marker.pose.position.y, radius))
        self.dynamic_obstacles = obstacles

    def on_metric_goal_distance(self, msg):
        self.metric_goal_distance = float(msg.data)

    def on_metric_min_obstacle(self, msg):
        self.metric_min_obstacle = float(msg.data)

    def fallback_goal_distance(self):
        if not self.has_pose or not self.has_goal:
            return -1.0
        return math.hypot(self.goal_x - self.x, self.goal_y - self.y)

    def nearest_obstacle_distance(self):
        if self.metric_min_obstacle >= 0.0:
            return self.metric_min_obstacle

        if not self.has_pose or not self.dynamic_obstacles:
            return -1.0

        nearest = None
        for ox, oy, radius in self.dynamic_obstacles:
            distance = math.hypot(ox - self.x, oy - self.y) - radius
            if nearest is None or distance < nearest:
                nearest = distance
        return nearest if nearest is not None else -1.0

    def front_obstacle_distance(self):
        if not self.has_pose:
            return -1.0

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
            if angle > math.radians(65.0):
                continue

            distance = math.hypot(x_r, y_r) - radius
            if nearest is None or distance < nearest:
                nearest = distance

        return nearest if nearest is not None else -1.0

    def boundary_distance_forward(self):
        if not self.has_pose:
            return -1.0

        safe_min_x = self.map_min_x + self.boundary_margin
        safe_max_x = self.map_max_x - self.boundary_margin
        safe_min_y = self.map_min_y + self.boundary_margin
        safe_max_y = self.map_max_y - self.boundary_margin

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        distances = []

        if cos_yaw > 1e-6:
            distances.append((safe_max_x - self.x) / cos_yaw)
        elif cos_yaw < -1e-6:
            distances.append((safe_min_x - self.x) / cos_yaw)

        if sin_yaw > 1e-6:
            distances.append((safe_max_y - self.y) / sin_yaw)
        elif sin_yaw < -1e-6:
            distances.append((safe_min_y - self.y) / sin_yaw)

        positive = [d for d in distances if d >= 0.0]
        return min(positive) if positive else -1.0

    def compute_progress(self, goal_distance):
        now = self.now_s()
        dt = max(1e-6, now - self.prev_time_s)

        if goal_distance >= 0.0 and self.prev_goal_distance is not None:
            self.progress_mps = (self.prev_goal_distance - goal_distance) / dt

        if goal_distance >= 0.0:
            self.prev_goal_distance = goal_distance

        self.prev_time_s = now

    def build_state(self):
        goal_distance = self.metric_goal_distance
        if goal_distance < 0.0:
            goal_distance = self.fallback_goal_distance()

        self.compute_progress(goal_distance)

        min_obstacle = self.nearest_obstacle_distance()
        front_obstacle = self.front_obstacle_distance()
        forward_boundary = self.boundary_distance_forward()

        return {
            "schema_version": 1,
            "raw_linear": round(self.raw_cmd.linear.x, 4),
            "raw_angular": round(self.raw_cmd.angular.z, 4),
            "has_pose": bool(self.has_pose),
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "yaw": round(self.yaw, 4),
            "goal_distance_m": round(goal_distance, 4),
            "progress_mps": round(self.progress_mps, 4),
            "min_obstacle_distance_m": round(min_obstacle, 4),
            "front_obstacle_distance_m": round(front_obstacle, 4),
            "forward_boundary_distance_m": round(forward_boundary, 4),
            "prev_linear_scale": round(self.prev_action["linear_scale"], 4),
            "prev_angular_scale": round(self.prev_action["angular_scale"], 4),
            "prev_angular_residual": round(self.prev_action["angular_residual"], 4),
        }

    def risk_scale(self, distance):
        if distance < 0.0:
            return 1.0

        if distance <= self.emergency_distance:
            return 0.0

        if distance <= self.warning_distance:
            ratio = (distance - self.emergency_distance) / max(
                1e-6,
                self.warning_distance - self.emergency_distance,
            )
            return self.clamp(0.20 + 0.45 * ratio, 0.0, 1.0)

        if distance <= self.caution_distance:
            ratio = (distance - self.warning_distance) / max(
                1e-6,
                self.caution_distance - self.warning_distance,
            )
            return self.clamp(0.65 + 0.35 * ratio, 0.0, 1.0)

        return 1.0

    def boundary_scale(self, distance):
        if distance < 0.0:
            return 1.0
        if distance <= 0.05:
            return 0.0
        if distance < self.boundary_caution_margin:
            return self.clamp(distance / max(self.boundary_caution_margin, 1e-6), 0.0, 1.0)
        return 1.0

    def compute_rule_bootstrap_action(self, state):
        obstacle_scale = self.risk_scale(state["front_obstacle_distance_m"])
        boundary_scale = self.boundary_scale(state["forward_boundary_distance_m"])
        linear_scale = min(obstacle_scale, boundary_scale)

        angular_scale = 1.0
        angular_residual = 0.0

        if linear_scale < 0.45:
            angular_scale = 0.85

        action = {
            "linear_scale": self.clamp(
                linear_scale,
                self.min_linear_scale,
                self.max_linear_scale,
            ),
            "angular_scale": self.clamp(
                angular_scale,
                self.min_angular_scale,
                self.max_angular_scale,
            ),
            "angular_residual": self.clamp(
                angular_residual,
                -self.max_angular_residual,
                self.max_angular_residual,
            ),
            "source": "rule_bootstrap",
        }
        return action

    def compute_reward(self, state, action, cmd):
        obstacle_distance = state["min_obstacle_distance_m"]
        front_distance = state["front_obstacle_distance_m"]
        boundary_distance = state["forward_boundary_distance_m"]

        reward = 0.4 * max(0.0, state["progress_mps"])
        reward += 0.15 * max(0.0, cmd.linear.x)

        if obstacle_distance >= 0.0:
            if obstacle_distance < self.emergency_distance:
                reward -= 5.0
            elif obstacle_distance < self.warning_distance:
                reward -= 1.5 * (self.warning_distance - obstacle_distance)

        if front_distance >= 0.0 and front_distance < self.warning_distance:
            reward -= 1.0 * (self.warning_distance - front_distance)

        if boundary_distance >= 0.0 and boundary_distance < self.boundary_caution_margin:
            reward -= 1.0 * (self.boundary_caution_margin - boundary_distance)

        action_delta = abs(action["linear_scale"] - self.prev_action["linear_scale"])
        action_delta += abs(action["angular_scale"] - self.prev_action["angular_scale"])
        reward -= 0.05 * action_delta

        if abs(cmd.linear.x) < 1e-4 and state["goal_distance_m"] > 0.5:
            reward -= 0.02

        return reward

    def apply_action(self, action):
        cmd = Twist()
        cmd.linear.x = self.raw_cmd.linear.x * action["linear_scale"]
        cmd.angular.z = (
            self.raw_cmd.angular.z * action["angular_scale"]
            + action["angular_residual"]
        )

        cmd.linear.x = self.clamp(cmd.linear.x, -self.max_linear, self.max_linear)
        cmd.angular.z = self.clamp(cmd.angular.z, -self.max_angular, self.max_angular)
        return cmd

    def publish_json(self, publisher, data):
        msg = String()
        msg.data = json.dumps(data, separators=(",", ":"))
        publisher.publish(msg)

    def update(self):
        age = (self.get_clock().now() - self.last_raw_time).nanoseconds * 1e-9

        if age > self.cmd_timeout:
            cmd = Twist()
            self.cmd_pub.publish(cmd)
            self.publish_json(
                self.status_pub,
                {
                    "status": "STOP_CMD_TIMEOUT",
                    "age_s": round(age, 3),
                    "training_enabled": self.training_enabled,
                },
            )
            return

        state = self.build_state()
        action = self.compute_rule_bootstrap_action(state)
        cmd = self.apply_action(action)
        reward = self.compute_reward(state, action, cmd)

        self.cmd_pub.publish(cmd)

        if self.publish_debug:
            self.publish_json(self.state_pub, state)
            self.publish_json(self.action_pub, action)
            reward_msg = Float32()
            reward_msg.data = float(reward)
            self.reward_pub.publish(reward_msg)
            self.publish_json(
                self.status_pub,
                {
                    "status": "BOOTSTRAP_POLICY_ACTIVE",
                    "action_mode": self.action_mode,
                    "training_enabled": self.training_enabled,
                    "reward": round(reward, 4),
                    "raw_linear": round(self.raw_cmd.linear.x, 4),
                    "raw_angular": round(self.raw_cmd.angular.z, 4),
                    "linear_scale": round(action["linear_scale"], 4),
                    "angular_scale": round(action["angular_scale"], 4),
                    "cmd_linear": round(cmd.linear.x, 4),
                    "cmd_angular": round(cmd.angular.z, 4),
                    "front_obstacle_distance_m": state["front_obstacle_distance_m"],
                    "forward_boundary_distance_m": state["forward_boundary_distance_m"],
                },
            )

        self.prev_action = {
            "linear_scale": float(action["linear_scale"]),
            "angular_scale": float(action["angular_scale"]),
            "angular_residual": float(action["angular_residual"]),
        }


def main():
    rclpy.init()
    node = OnlineSacSafetyPolicy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
