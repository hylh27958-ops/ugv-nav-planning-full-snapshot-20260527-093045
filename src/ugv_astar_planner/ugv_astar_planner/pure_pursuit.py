import json
import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import String


class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit")

        self.declare_parameter("update_period", 0.1)
        self.declare_parameter("lookahead", 0.7)
        self.declare_parameter("max_linear", 0.45)
        self.declare_parameter("max_angular", 1.2)
        self.declare_parameter("goal_tolerance", 0.25)
        self.declare_parameter("min_linear", 0.08)

        self.declare_parameter("use_sac_adaptation", True)
        self.declare_parameter("adaptation_timeout", 1.0)
        self.declare_parameter("min_speed_scale", 0.40)
        self.declare_parameter("max_speed_scale", 1.00)
        self.declare_parameter("min_lookahead_scale", 0.60)
        self.declare_parameter("max_lookahead_scale", 1.50)
        self.declare_parameter("min_lookahead", 0.35)
        self.declare_parameter("max_lookahead", 1.20)
        self.declare_parameter("angular_slowdown_gain", 0.45)

        self.update_period = float(self.get_parameter("update_period").value)
        self.base_lookahead = float(self.get_parameter("lookahead").value)
        self.base_max_linear = float(self.get_parameter("max_linear").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.min_linear = float(self.get_parameter("min_linear").value)

        self.use_sac_adaptation = bool(self.get_parameter("use_sac_adaptation").value)
        self.adaptation_timeout = float(self.get_parameter("adaptation_timeout").value)
        self.min_speed_scale = float(self.get_parameter("min_speed_scale").value)
        self.max_speed_scale = float(self.get_parameter("max_speed_scale").value)
        self.min_lookahead_scale = float(self.get_parameter("min_lookahead_scale").value)
        self.max_lookahead_scale = float(self.get_parameter("max_lookahead_scale").value)
        self.min_lookahead = float(self.get_parameter("min_lookahead").value)
        self.max_lookahead = float(self.get_parameter("max_lookahead").value)
        self.angular_slowdown_gain = float(self.get_parameter("angular_slowdown_gain").value)

        self.create_subscription(Path, "/local_trajectory", self.on_path, 10)
        self.create_subscription(PoseStamped, "/robot_pose", self.on_robot_pose, 10)
        self.create_subscription(PoseStamped, "/astar_goal", self.on_goal_pose, 10)
        self.create_subscription(String, "/sac/adaptation", self.on_sac_adaptation, 10)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel_raw", 10)
        self.status_pub = self.create_publisher(String, "/pure_pursuit/status", 10)

        self.path = []
        self.has_robot_pose = False
        self.has_goal = False

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.goal_x = 0.0
        self.goal_y = 0.0

        self.speed_scale = 1.0
        self.lookahead_scale = 1.0
        self.obstacle_caution = 1.0
        self.local_horizon_scale = 1.0
        self.last_adaptation_time = 0.0
        self.adaptation_mode = "none"

        self.timer = self.create_timer(self.update_period, self.on_timer)

        self.get_logger().info("SAC-adaptive Pure Pursuit started.")

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def yaw_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def on_path(self, msg):
        self.path = [
            (pose.pose.position.x, pose.pose.position.y)
            for pose in msg.poses
        ]

    def on_robot_pose(self, msg):
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y
        self.robot_yaw = self.yaw_from_quaternion(msg.pose.orientation)
        self.has_robot_pose = True

    def on_goal_pose(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.has_goal = True

    def on_sac_adaptation(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Failed to parse /sac/adaptation JSON.")
            return

        action = data.get("action", data)

        speed_scale = action.get("speed_scale", data.get("speed_scale", 1.0))
        lookahead_scale = action.get("lookahead_scale", data.get("lookahead_scale", 1.0))
        obstacle_caution = action.get("obstacle_caution", data.get("obstacle_caution", 1.0))
        local_horizon_scale = action.get(
            "local_horizon_scale",
            data.get("local_horizon_scale", 1.0),
        )

        try:
            self.speed_scale = self.clamp(
                float(speed_scale), self.min_speed_scale, self.max_speed_scale
            )
            self.lookahead_scale = self.clamp(
                float(lookahead_scale),
                self.min_lookahead_scale,
                self.max_lookahead_scale,
            )
            self.obstacle_caution = max(0.0, float(obstacle_caution))
            self.local_horizon_scale = max(0.0, float(local_horizon_scale))
            self.adaptation_mode = data.get("mode", "unknown")
            self.last_adaptation_time = time.time()
        except (TypeError, ValueError):
            self.get_logger().warn("Invalid SAC adaptation values.")

    def adaptation_is_fresh(self):
        if not self.use_sac_adaptation:
            return False

        if self.last_adaptation_time <= 0.0:
            return False

        return (time.time() - self.last_adaptation_time) <= self.adaptation_timeout

    def effective_scales(self):
        if self.adaptation_is_fresh():
            speed_scale = self.speed_scale
            lookahead_scale = self.lookahead_scale
            source = "sac"
        else:
            speed_scale = 1.0
            lookahead_scale = 1.0
            source = "default"

        effective_lookahead = self.clamp(
            self.base_lookahead * lookahead_scale,
            self.min_lookahead,
            self.max_lookahead,
        )
        effective_max_linear = self.clamp(
            self.base_max_linear * speed_scale,
            0.0,
            self.base_max_linear,
        )

        return effective_lookahead, effective_max_linear, speed_scale, lookahead_scale, source

    def publish_zero(self, status):
        cmd = Twist()
        self.cmd_pub.publish(cmd)
        self.publish_status(status, 0.0, 0.0, 0.0, 0.0, "none")

    def publish_status(
        self,
        navigation_state,
        effective_lookahead,
        effective_max_linear,
        linear,
        angular,
        adaptation_source,
    ):
        msg = String()
        msg.data = json.dumps(
            {
                "navigation_state": navigation_state,
                "adaptation_source": adaptation_source,
                "adaptation_mode": self.adaptation_mode,
                "speed_scale": round(self.speed_scale, 3),
                "lookahead_scale": round(self.lookahead_scale, 3),
                "effective_lookahead": round(effective_lookahead, 3),
                "effective_max_linear": round(effective_max_linear, 3),
                "cmd_linear": round(linear, 3),
                "cmd_angular": round(angular, 3),
            },
            separators=(",", ":"),
        )
        self.status_pub.publish(msg)

    def goal_distance(self):
        if not self.has_goal or not self.has_robot_pose:
            return None
        return math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)

    def nearest_path_index(self):
        nearest_idx = 0
        nearest_dist = None

        for i, (x, y) in enumerate(self.path):
            dist = math.hypot(x - self.robot_x, y - self.robot_y)
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = i

        return nearest_idx

    def select_lookahead_point(self, effective_lookahead):
        if not self.path:
            return None

        start_idx = self.nearest_path_index()

        for i in range(start_idx, len(self.path)):
            x, y = self.path[i]
            dist = math.hypot(x - self.robot_x, y - self.robot_y)
            if dist >= effective_lookahead:
                return x, y

        return self.path[-1]

    def on_timer(self):
        if not self.has_robot_pose:
            self.publish_zero("WAITING_FOR_ROBOT_POSE")
            return

        if len(self.path) < 2:
            self.publish_zero("WAITING_FOR_LOCAL_TRAJECTORY")
            return

        effective_lookahead, effective_max_linear, speed_scale, lookahead_scale, source = (
            self.effective_scales()
        )

        goal_dist = self.goal_distance()
        if goal_dist is not None and goal_dist <= self.goal_tolerance:
            self.publish_zero("REACHED")
            return

        target = self.select_lookahead_point(effective_lookahead)
        if target is None:
            self.publish_zero("NO_LOOKAHEAD_POINT")
            return

        tx, ty = target
        dx = tx - self.robot_x
        dy = ty - self.robot_y
        target_dist = math.hypot(dx, dy)

        if target_dist < 1e-6:
            self.publish_zero("TARGET_TOO_CLOSE")
            return

        target_angle = math.atan2(dy, dx)
        alpha = self.normalize_angle(target_angle - self.robot_yaw)

        curvature = 2.0 * math.sin(alpha) / max(target_dist, 1e-6)

        turn_ratio = min(abs(alpha) / math.pi, 1.0)
        turn_scale = self.clamp(
            1.0 - self.angular_slowdown_gain * turn_ratio,
            0.25,
            1.0,
        )

        linear = effective_max_linear * turn_scale

        if goal_dist is not None and goal_dist < effective_lookahead:
            linear *= self.clamp(goal_dist / effective_lookahead, 0.15, 1.0)

        if linear > 0.0:
            linear = max(self.min_linear, linear)

        angular = linear * curvature
        angular = self.clamp(angular, -self.max_angular, self.max_angular)

        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_pub.publish(cmd)

        self.publish_status(
            "TRACKING",
            effective_lookahead,
            effective_max_linear,
            linear,
            angular,
            source,
        )


def main():
    rclpy.init()
    node = PurePursuit()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
