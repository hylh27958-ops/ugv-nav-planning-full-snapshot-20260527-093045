import json
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray


class MinimalSafetyShield(Node):
    def __init__(self):
        super().__init__("minimal_safety_shield")

        self.declare_parameter("update_period", 0.05)
        self.declare_parameter("max_linear", 0.45)
        self.declare_parameter("max_angular", 1.2)
        self.declare_parameter("cmd_timeout", 0.5)
        self.declare_parameter("emergency_distance", 0.18)
        self.declare_parameter("front_angle_deg", 65.0)

        self.declare_parameter("enable_boundary_guard", True)
        self.declare_parameter("map_min_x", 0.0)
        self.declare_parameter("map_max_x", 14.0)
        self.declare_parameter("map_min_y", 0.0)
        self.declare_parameter("map_max_y", 9.0)
        self.declare_parameter("boundary_margin", 0.25)

        self.update_period = float(self.get_parameter("update_period").value)
        self.max_linear = float(self.get_parameter("max_linear").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.cmd_timeout = float(self.get_parameter("cmd_timeout").value)
        self.emergency_distance = float(self.get_parameter("emergency_distance").value)
        self.front_angle = math.radians(float(self.get_parameter("front_angle_deg").value))

        self.enable_boundary_guard = bool(self.get_parameter("enable_boundary_guard").value)
        self.map_min_x = float(self.get_parameter("map_min_x").value)
        self.map_max_x = float(self.get_parameter("map_max_x").value)
        self.map_min_y = float(self.get_parameter("map_min_y").value)
        self.map_max_y = float(self.get_parameter("map_max_y").value)
        self.boundary_margin = float(self.get_parameter("boundary_margin").value)

        self.cmd_in = Twist()
        self.last_cmd_time = self.get_clock().now()

        self.has_pose = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.dynamic_obstacles = []

        self.create_subscription(Twist, "/cmd_vel_sac", self.on_cmd_sac, 10)
        self.create_subscription(PoseStamped, "/robot_pose", self.on_robot_pose, 10)
        self.create_subscription(MarkerArray, "/dynamic_obstacles", self.on_dynamic_obstacles, 10)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.status_pub = self.create_publisher(String, "/minimal_safety_status", 10)

        self.timer = self.create_timer(self.update_period, self.update)
        self.get_logger().info("Minimal safety shield started: /cmd_vel_sac -> /cmd_vel")

    def clamp(self, value, lo, hi):
        return max(lo, min(hi, value))

    def yaw_from_quat(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def on_cmd_sac(self, msg):
        self.cmd_in = msg
        self.last_cmd_time = self.get_clock().now()

    def on_robot_pose(self, msg):
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.yaw = self.yaw_from_quat(msg.pose.orientation)
        self.has_pose = True

    def on_dynamic_obstacles(self, msg):
        obstacles = []
        for marker in msg.markers:
            if marker.action == marker.DELETE:
                continue
            radius = max(marker.scale.x, marker.scale.y) * 0.5
            obstacles.append((marker.pose.position.x, marker.pose.position.y, radius))
        self.dynamic_obstacles = obstacles

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

    def moving_toward_boundary(self, linear):
        if not self.enable_boundary_guard or not self.has_pose:
            return False, None

        if linear <= 1e-6:
            return False, None

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
            return False, None

        nearest = min(distances)
        if nearest <= 0.02:
            return True, nearest

        return False, nearest

    def publish_status(self, data):
        msg = String()
        msg.data = json.dumps(data, separators=(",", ":"))
        self.status_pub.publish(msg)

    def update(self):
        now = self.get_clock().now()
        age = (now - self.last_cmd_time).nanoseconds * 1e-9

        out = Twist()
        status = "OK"
        reason = ""

        if age > self.cmd_timeout:
            status = "STOP"
            reason = "cmd_vel_sac timeout"
        else:
            out.linear.x = self.clamp(self.cmd_in.linear.x, -self.max_linear, self.max_linear)
            out.angular.z = self.clamp(self.cmd_in.angular.z, -self.max_angular, self.max_angular)

            nearest_front = self.nearest_front_obstacle_distance()
            if (
                nearest_front is not None
                and nearest_front < self.emergency_distance
                and out.linear.x > 0.0
            ):
                out = Twist()
                status = "STOP"
                reason = f"emergency obstacle {nearest_front:.3f} m"

            boundary_stop, boundary_distance = self.moving_toward_boundary(out.linear.x)
            if boundary_stop:
                out.linear.x = 0.0
                status = "STOP"
                reason = f"map boundary {boundary_distance:.3f} m"

        self.cmd_pub.publish(out)
        self.publish_status(
            {
                "status": status,
                "reason": reason,
                "cmd_age_s": round(age, 3),
                "in_linear": round(self.cmd_in.linear.x, 4),
                "in_angular": round(self.cmd_in.angular.z, 4),
                "out_linear": round(out.linear.x, 4),
                "out_angular": round(out.angular.z, 4),
            }
        )


def main():
    rclpy.init()
    node = MinimalSafetyShield()
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
