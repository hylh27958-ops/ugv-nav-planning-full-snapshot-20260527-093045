import json
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String


class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit")

        self.declare_parameter("update_period", 0.1)
        self.declare_parameter("lookahead", 0.7)
        self.declare_parameter("max_linear", 0.45)
        self.declare_parameter("max_angular", 1.2)
        self.declare_parameter("goal_tolerance", 0.25)

        self.update_period = float(self.get_parameter("update_period").value)
        self.lookahead = float(self.get_parameter("lookahead").value)
        self.max_linear = float(self.get_parameter("max_linear").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)

        self.max_linear_scale = 1.0
        self.lookahead_scale = 1.0

        self.create_subscription(Path, "/local_trajectory", self.on_path, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(String, "/sac/adaptation", self.on_adaptation, 10)

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel_raw", 10)

        self.path = []
        self.has_odom = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.timer = self.create_timer(self.update_period, self.control_loop)
        self.get_logger().info("Pure Pursuit started. Tracking /local_trajectory.")

    def on_adaptation(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        self.max_linear_scale = float(data.get("max_linear_scale", 1.0))
        self.lookahead_scale = float(data.get("lookahead_scale", 1.0))

    def effective_max_linear(self):
        return max(0.05, self.max_linear * self.max_linear_scale)

    def effective_lookahead(self):
        return max(0.25, self.lookahead * self.lookahead_scale)

    def on_path(self, msg):
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def on_odom(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        self.yaw = self.yaw_from_quat(q.x, q.y, q.z, q.w)
        self.has_odom = True

    def yaw_from_quat(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def clamp(self, value, lo, hi):
        return max(lo, min(hi, value))

    def distance(self, p):
        return math.hypot(p[0] - self.x, p[1] - self.y)

    def find_target_point(self):
        if not self.path:
            return None

        lookahead = self.effective_lookahead()
        nearest_i = min(range(len(self.path)), key=lambda i: self.distance(self.path[i]))

        for i in range(nearest_i, len(self.path)):
            if self.distance(self.path[i]) >= lookahead:
                return self.path[i]

        return self.path[-1]

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def control_loop(self):
        if not self.has_odom or len(self.path) < 2:
            self.publish_stop()
            return

        goal = self.path[-1]

        if self.distance(goal) < self.goal_tolerance:
            self.publish_stop()
            return

        target = self.find_target_point()
        if target is None:
            self.publish_stop()
            return

        dx = target[0] - self.x
        dy = target[1] - self.y

        x_r = math.cos(self.yaw) * dx + math.sin(self.yaw) * dy
        y_r = -math.sin(self.yaw) * dx + math.cos(self.yaw) * dy

        if x_r < 0.05:
            cmd = Twist()
            cmd.angular.z = 0.6 if y_r > 0.0 else -0.6
            self.cmd_pub.publish(cmd)
            return

        ld2 = max(x_r * x_r + y_r * y_r, 1e-6)
        curvature = 2.0 * y_r / ld2

        linear = self.effective_max_linear()
        angular = self.clamp(curvature * linear, -self.max_angular, self.max_angular)

        if abs(angular) > 0.8:
            linear *= 0.5

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
