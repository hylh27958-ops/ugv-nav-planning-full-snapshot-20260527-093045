import json
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray


class SafetyFilter(Node):
    def __init__(self):
        super().__init__("safety_filter")

        self.declare_parameter("update_period", 0.05)
        self.declare_parameter("max_linear", 0.45)
        self.declare_parameter("max_angular", 1.2)
        self.declare_parameter("max_linear_accel", 0.45)
        self.declare_parameter("max_angular_accel", 1.8)
        self.declare_parameter("cmd_timeout", 0.5)
        self.declare_parameter("front_angle_deg", 60.0)
        self.declare_parameter("stop_distance", 0.45)
        self.declare_parameter("slow_distance", 1.2)
        self.declare_parameter("enable_boundary_guard", True)
        self.declare_parameter("map_min_x", 0.0)
        self.declare_parameter("map_max_x", 14.0)
        self.declare_parameter("map_min_y", 0.0)
        self.declare_parameter("map_max_y", 9.0)
        self.declare_parameter("boundary_margin", 0.25)
        self.declare_parameter("boundary_slow_margin", 0.45)
        self.declare_parameter("goal_switch_stop_s", 0.5)
        self.declare_parameter("goal_change_distance", 0.2)

        self.update_period = float(self.get_parameter("update_period").value)
        self.max_linear = float(self.get_parameter("max_linear").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.max_linear_accel = float(self.get_parameter("max_linear_accel").value)
        self.max_angular_accel = float(self.get_parameter("max_angular_accel").value)
        self.cmd_timeout = float(self.get_parameter("cmd_timeout").value)
        self.front_angle = math.radians(float(self.get_parameter("front_angle_deg").value))
        self.stop_distance = float(self.get_parameter("stop_distance").value)
        self.slow_distance = float(self.get_parameter("slow_distance").value)
        self.enable_boundary_guard = bool(self.get_parameter("enable_boundary_guard").value)
        self.map_min_x = float(self.get_parameter("map_min_x").value)
        self.map_max_x = float(self.get_parameter("map_max_x").value)
        self.map_min_y = float(self.get_parameter("map_min_y").value)
        self.map_max_y = float(self.get_parameter("map_max_y").value)
        self.boundary_margin = float(self.get_parameter("boundary_margin").value)
        self.boundary_slow_margin = float(self.get_parameter("boundary_slow_margin").value)
        self.goal_switch_stop_s = float(self.get_parameter("goal_switch_stop_s").value)
        self.goal_change_distance = float(self.get_parameter("goal_change_distance").value)

        self.max_linear_scale = 1.0
        self.stop_distance_scale = 1.0
        self.slow_distance_scale = 1.0

        self.create_subscription(Twist, "/cmd_vel_raw", self.on_cmd_raw, 10)
        self.create_subscription(PoseStamped, "/robot_pose", self.on_robot_pose, 10)
        self.create_subscription(PoseStamped, "/astar_goal", self.on_goal_pose, 10)
        self.create_subscription(MarkerArray, "/dynamic_obstacles", self.on_dynamic_obstacles, 10)
        self.create_subscription(String, "/sac/adaptation", self.on_adaptation, 10)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.status_pub = self.create_publisher(String, "/safety_status", 10)

        self.raw_cmd = Twist()
        self.last_raw_time = self.get_clock().now()
        self.last_out = Twist()
        self.last_update_time = self.get_clock().now()

        self.has_pose = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.dynamic_obstacles = []
        self.last_goal = None
        self.goal_switch_hold_until_s = 0.0

        self.timer = self.create_timer(self.update_period, self.update)
        self.get_logger().info("Safety filter started. /cmd_vel_raw -> /cmd_vel")

    def on_adaptation(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        self.max_linear_scale = float(data.get("max_linear_scale", 1.0))
        self.stop_distance_scale = float(data.get("stop_distance_scale", 1.0))
        self.slow_distance_scale = float(data.get("slow_distance_scale", 1.0))

    def effective_max_linear(self):
        return max(0.05, self.max_linear * self.max_linear_scale)

    def effective_stop_distance(self):
        return max(0.1, self.stop_distance * self.stop_distance_scale)

    def effective_slow_distance(self):
        stop = self.effective_stop_distance()
        return max(stop + 0.1, self.slow_distance * self.slow_distance_scale)

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_cmd_raw(self, msg):
        self.raw_cmd = msg
        self.last_raw_time = self.get_clock().now()

    def on_robot_pose(self, msg):
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y

        q = msg.pose.orientation
        self.yaw = self.yaw_from_quat(q.x, q.y, q.z, q.w)
        self.has_pose = True

    def on_goal_pose(self, msg):
        new_goal = (msg.pose.position.x, msg.pose.position.y)

        if self.last_goal is not None:
            moved = math.hypot(
                new_goal[0] - self.last_goal[0],
                new_goal[1] - self.last_goal[1],
            )
            if moved > self.goal_change_distance:
                self.goal_switch_hold_until_s = self.now_s() + self.goal_switch_stop_s
                self.raw_cmd = Twist()
                self.last_out = Twist()
                self.last_raw_time = self.get_clock().now()

        self.last_goal = new_goal

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

            distance = math.hypot(x_r, y_r) - radius

            if nearest is None or distance < nearest:
                nearest = distance

        return nearest

    def boundary_linear_scale(self, linear):
        if not self.enable_boundary_guard or not self.has_pose:
            return 1.0, None

        if abs(linear) < 1e-6:
            return 1.0, None

        safe_min_x = self.map_min_x + self.boundary_margin
        safe_max_x = self.map_max_x - self.boundary_margin
        safe_min_y = self.map_min_y + self.boundary_margin
        safe_max_y = self.map_max_y - self.boundary_margin

        vx = math.cos(self.yaw) * linear
        vy = math.sin(self.yaw) * linear

        distances = []

        if vx < -1e-6:
            distances.append(self.x - safe_min_x)
        elif vx > 1e-6:
            distances.append(safe_max_x - self.x)

        if vy < -1e-6:
            distances.append(self.y - safe_min_y)
        elif vy > 1e-6:
            distances.append(safe_max_y - self.y)

        if not distances:
            return 1.0, None

        nearest = min(distances)

        if nearest <= 0.02:
            return 0.0, f"map boundary {nearest:.2f} m"

        if nearest < self.boundary_slow_margin:
            scale = self.clamp(nearest / max(self.boundary_slow_margin, 1e-6), 0.0, 1.0)
            return scale, f"map boundary {nearest:.2f} m"

        return 1.0, None

    def publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def update(self):
        now = self.get_clock().now()
        dt = (now - self.last_update_time).nanoseconds * 1e-9
        self.last_update_time = now

        if dt <= 0.0 or dt > 0.2:
            dt = self.update_period

        age = (now - self.last_raw_time).nanoseconds * 1e-9
        target = Twist()
        status = "OK"

        max_linear = self.effective_max_linear()
        stop_distance = self.effective_stop_distance()
        slow_distance = self.effective_slow_distance()

        if self.now_s() < self.goal_switch_hold_until_s:
            self.raw_cmd = Twist()
            self.last_out = Twist()
            self.cmd_pub.publish(Twist())
            self.publish_status("STOP: goal switch")
            return

        if age > self.cmd_timeout:
            status = "STOP: cmd_vel_raw timeout"
        else:
            target.linear.x = self.clamp(self.raw_cmd.linear.x, -max_linear, max_linear)
            target.angular.z = self.clamp(self.raw_cmd.angular.z, -self.max_angular, self.max_angular)

            nearest = self.nearest_front_obstacle_distance()

            if nearest is not None:
                if nearest < stop_distance and target.linear.x > 0.0:
                    target = Twist()
                    status = f"STOP: obstacle {nearest:.2f} m"
                elif nearest < slow_distance and target.linear.x > 0.0:
                    scale = (nearest - stop_distance) / max(
                        slow_distance - stop_distance,
                        1e-6,
                    )
                    scale = self.clamp(scale, 0.0, 1.0)
                    target.linear.x *= scale
                    status = f"SLOW: obstacle {nearest:.2f} m"

            boundary_scale, boundary_reason = self.boundary_linear_scale(target.linear.x)
            if boundary_reason is not None:
                if boundary_scale <= 0.0:
                    target.linear.x = 0.0
                    status = f"STOP: {boundary_reason}"
                else:
                    target.linear.x *= boundary_scale
                    status = f"SLOW: {boundary_reason}"

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
        self.publish_status(status)


def main():
    rclpy.init()
    node = SafetyFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
