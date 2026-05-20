import json
import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import Float32, String
from visualization_msgs.msg import MarkerArray


class SacAdapter(Node):
    def __init__(self):
        super().__init__("sac_adapter")

        self.declare_parameter("update_period", 0.2)
        self.declare_parameter("mode", "b4_v2_proactive_rule_policy")
        self.declare_parameter("caution_distance", 1.40)
        self.declare_parameter("warning_distance", 0.75)
        self.declare_parameter("emergency_distance", 0.40)
        self.declare_parameter("speed_floor", 0.62)
        self.declare_parameter("speed_ceiling", 1.0)
        self.declare_parameter("lookahead_ceiling", 1.28)
        self.declare_parameter("local_horizon_ceiling", 1.45)
        self.declare_parameter("obstacle_caution_ceiling", 1.70)
        self.declare_parameter("rise_alpha", 0.55)
        self.declare_parameter("fall_alpha", 0.22)

        self.update_period = float(self.get_parameter("update_period").value)
        self.mode = str(self.get_parameter("mode").value)

        self.caution_distance = float(self.get_parameter("caution_distance").value)
        self.warning_distance = float(self.get_parameter("warning_distance").value)
        self.emergency_distance = float(self.get_parameter("emergency_distance").value)

        self.speed_floor = float(self.get_parameter("speed_floor").value)
        self.speed_ceiling = float(self.get_parameter("speed_ceiling").value)
        self.lookahead_ceiling = float(self.get_parameter("lookahead_ceiling").value)
        self.local_horizon_ceiling = float(self.get_parameter("local_horizon_ceiling").value)
        self.obstacle_caution_ceiling = float(self.get_parameter("obstacle_caution_ceiling").value)

        self.rise_alpha = float(self.get_parameter("rise_alpha").value)
        self.fall_alpha = float(self.get_parameter("fall_alpha").value)

        self.robot_pose = None
        self.goal_pose = None
        self.dynamic_obstacles = []
        self.global_path = []
        self.local_path = []

        self.goal_distance_m = -1.0
        self.min_obstacle_distance_m = -1.0
        self.safety_scale = 1.0
        self.raw_linear_mps = 0.0
        self.safe_linear_mps = 0.0

        self.speed_scale = 1.0
        self.lookahead_scale = 1.0
        self.local_horizon_scale = 1.0
        self.obstacle_caution = 1.0

        self.prev_goal_distance = None
        self.prev_stamp = time.time()
        self.progress_mps = 0.0

        self.create_subscription(PoseStamped, "/robot_pose", self.on_robot_pose, 10)
        self.create_subscription(PoseStamped, "/astar_goal", self.on_goal_pose, 10)
        self.create_subscription(MarkerArray, "/dynamic_obstacles", self.on_dynamic_obstacles, 10)
        self.create_subscription(Path, "/astar_path", self.on_global_path, 10)
        self.create_subscription(Path, "/local_trajectory", self.on_local_path, 10)
        self.create_subscription(Twist, "/cmd_vel_raw", self.on_cmd_raw, 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd_safe, 10)

        self.create_subscription(Float32, "/metrics/goal_distance", self.on_metric_goal_distance, 10)
        self.create_subscription(Float32, "/metrics/min_dynamic_obstacle_distance", self.on_metric_min_obstacle, 10)
        self.create_subscription(Float32, "/metrics/safety_scale", self.on_metric_safety_scale, 10)

        self.adaptation_pub = self.create_publisher(String, "/sac/adaptation", 10)
        self.state_pub = self.create_publisher(String, "/sac/state", 10)
        self.reward_pub = self.create_publisher(Float32, "/sac/reward", 10)

        self.timer = self.create_timer(self.update_period, self.on_timer)
        self.get_logger().info("B4-v2 proactive SAC adapter started.")

    def on_robot_pose(self, msg):
        self.robot_pose = msg

    def on_goal_pose(self, msg):
        self.goal_pose = msg

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

    def on_global_path(self, msg):
        self.global_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def on_local_path(self, msg):
        self.local_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def on_cmd_raw(self, msg):
        self.raw_linear_mps = msg.linear.x

    def on_cmd_safe(self, msg):
        self.safe_linear_mps = msg.linear.x

    def on_metric_goal_distance(self, msg):
        self.goal_distance_m = float(msg.data)

    def on_metric_min_obstacle(self, msg):
        self.min_obstacle_distance_m = float(msg.data)

    def on_metric_safety_scale(self, msg):
        self.safety_scale = max(0.0, min(1.0, float(msg.data)))

    def path_length(self, points):
        if len(points) < 2:
            return 0.0
        return sum(
            math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
            for i in range(len(points) - 1)
        )

    def fallback_goal_distance(self):
        if self.robot_pose is None or self.goal_pose is None:
            return -1.0
        rx = self.robot_pose.pose.position.x
        ry = self.robot_pose.pose.position.y
        gx = self.goal_pose.pose.position.x
        gy = self.goal_pose.pose.position.y
        return math.hypot(gx - rx, gy - ry)

    def fallback_min_obstacle_distance(self):
        if self.robot_pose is None or not self.dynamic_obstacles:
            return -1.0
        rx = self.robot_pose.pose.position.x
        ry = self.robot_pose.pose.position.y
        nearest = None
        for ox, oy, radius in self.dynamic_obstacles:
            d = math.hypot(ox - rx, oy - ry) - radius
            if nearest is None or d < nearest:
                nearest = d
        return nearest if nearest is not None else -1.0

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def blend(self, old, target):
        alpha = self.rise_alpha if abs(target - 1.0) > abs(old - 1.0) else self.fall_alpha
        return old + alpha * (target - old)

    def compute_action(self, obstacle_distance):
        if obstacle_distance < 0.0:
            return 1.0, 1.0, 1.0, 1.0, 0.0

        if obstacle_distance >= self.caution_distance:
            risk = 0.0
            speed = 1.0
            lookahead = 1.0
            horizon = 1.0
            caution = 1.0
        elif obstacle_distance >= self.warning_distance:
            r = (self.caution_distance - obstacle_distance) / max(1e-6, self.caution_distance - self.warning_distance)
            risk = 0.35 * r
            speed = 1.0 - 0.10 * r
            lookahead = 1.0 + 0.20 * r
            horizon = 1.0 + 0.25 * r
            caution = 1.0 + 0.35 * r
        elif obstacle_distance >= self.emergency_distance:
            r = (self.warning_distance - obstacle_distance) / max(1e-6, self.warning_distance - self.emergency_distance)
            risk = 0.35 + 0.45 * r
            speed = 0.90 - 0.18 * r
            lookahead = 1.20 + 0.08 * r
            horizon = 1.25 + 0.18 * r
            caution = 1.35 + 0.30 * r
        else:
            risk = 1.0
            speed = self.speed_floor
            lookahead = 1.12
            horizon = self.local_horizon_ceiling
            caution = self.obstacle_caution_ceiling

        if self.safety_scale < 0.65:
            speed = min(speed, 0.82)
            horizon = max(horizon, 1.30)
            caution = max(caution, 1.45)

        speed = self.clamp(speed, self.speed_floor, self.speed_ceiling)
        lookahead = self.clamp(lookahead, 0.95, self.lookahead_ceiling)
        horizon = self.clamp(horizon, 1.0, self.local_horizon_ceiling)
        caution = self.clamp(caution, 1.0, self.obstacle_caution_ceiling)

        return speed, lookahead, horizon, caution, risk

    def compute_progress(self, goal_distance):
        now = time.time()
        dt = max(1e-6, now - self.prev_stamp)

        if goal_distance >= 0.0 and self.prev_goal_distance is not None:
            self.progress_mps = (self.prev_goal_distance - goal_distance) / dt

        if goal_distance >= 0.0:
            self.prev_goal_distance = goal_distance

        self.prev_stamp = now

    def publish_string(self, publisher, data):
        msg = String()
        msg.data = data
        publisher.publish(msg)

    def publish_reward(self, reward):
        msg = Float32()
        msg.data = float(reward)
        self.reward_pub.publish(msg)

    def on_timer(self):
        goal_distance = self.goal_distance_m
        if goal_distance < 0.0:
            goal_distance = self.fallback_goal_distance()

        obstacle_distance = self.min_obstacle_distance_m
        if obstacle_distance < 0.0:
            obstacle_distance = self.fallback_min_obstacle_distance()

        self.compute_progress(goal_distance)

        target_speed, target_lookahead, target_horizon, target_caution, risk = self.compute_action(obstacle_distance)

        self.speed_scale = self.blend(self.speed_scale, target_speed)
        self.lookahead_scale = self.blend(self.lookahead_scale, target_lookahead)
        self.local_horizon_scale = self.blend(self.local_horizon_scale, target_horizon)
        self.obstacle_caution = self.blend(self.obstacle_caution, target_caution)

        global_len = self.path_length(self.global_path)
        local_len = self.path_length(self.local_path)

        reward = 0.2 * self.safe_linear_mps
        reward += 0.3 * max(0.0, self.progress_mps)
        reward -= 0.35 * risk
        reward -= 0.20 * max(0.0, 1.0 - self.safety_scale)
        if goal_distance >= 0.0:
            reward -= 0.01 * goal_distance
            if goal_distance < 0.35:
                reward += 1.0

        state = {
            "schema_version": 1,
            "goal_distance_m": round(goal_distance, 3),
            "min_dynamic_obstacle_distance_m": round(obstacle_distance, 3),
            "safety_scale": round(self.safety_scale, 3),
            "raw_linear_mps": round(self.raw_linear_mps, 3),
            "safe_linear_mps": round(self.safe_linear_mps, 3),
            "progress_mps": round(self.progress_mps, 3),
            "global_path_length_m": round(global_len, 3),
            "local_trajectory_length_m": round(local_len, 3),
        }

        action = {
            "speed_scale": round(self.speed_scale, 3),
            "lookahead_scale": round(self.lookahead_scale, 3),
            "local_horizon_scale": round(self.local_horizon_scale, 3),
            "obstacle_caution": round(self.obstacle_caution, 3),
        }

        adaptation = {
            "schema_version": 1,
            "mode": self.mode,
            "policy_type": "rl_parameter_interface",
            "stamp": time.time(),
            "state": state,
            "action": action,
            "risk": round(risk, 3),
            "reward": round(reward, 4),
            "speed_scale": action["speed_scale"],
            "lookahead_scale": action["lookahead_scale"],
            "local_horizon_scale": action["local_horizon_scale"],
            "obstacle_caution": action["obstacle_caution"],
        }

        self.publish_string(self.state_pub, json.dumps(state, separators=(",", ":")))
        self.publish_string(self.adaptation_pub, json.dumps(adaptation, separators=(",", ":")))
        self.publish_reward(reward)


def main():
    rclpy.init()
    node = SacAdapter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
