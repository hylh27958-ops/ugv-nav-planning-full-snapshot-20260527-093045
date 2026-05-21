import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


class FakeBase(Node):
    def __init__(self):
        super().__init__("fake_base")

        self.declare_parameter("initial_x", 0.8)
        self.declare_parameter("initial_y", 0.8)
        self.declare_parameter("initial_yaw", 0.0)
        self.declare_parameter("update_period", 0.05)
        self.declare_parameter("cmd_timeout", 0.5)
        self.declare_parameter("enforce_map_bounds", True)
        self.declare_parameter("map_min_x", 0.0)
        self.declare_parameter("map_max_x", 14.0)
        self.declare_parameter("map_min_y", 0.0)
        self.declare_parameter("map_max_y", 9.0)
        self.declare_parameter("boundary_margin", 0.25)

        self.x = float(self.get_parameter("initial_x").value)
        self.y = float(self.get_parameter("initial_y").value)
        self.yaw = float(self.get_parameter("initial_yaw").value)
        self.update_period = float(self.get_parameter("update_period").value)
        self.cmd_timeout = float(self.get_parameter("cmd_timeout").value)
        self.enforce_map_bounds = bool(self.get_parameter("enforce_map_bounds").value)
        self.map_min_x = float(self.get_parameter("map_min_x").value)
        self.map_max_x = float(self.get_parameter("map_max_x").value)
        self.map_min_y = float(self.get_parameter("map_min_y").value)
        self.map_max_y = float(self.get_parameter("map_max_y").value)
        self.boundary_margin = float(self.get_parameter("boundary_margin").value)

        self.create_subscription(Twist, "/cmd_vel", self.on_cmd_vel, 10)

        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "/robot_pose", 10)
        self.start_pub = self.create_publisher(PoseStamped, "/start_pose", 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.v = 0.0
        self.w = 0.0
        self.last_cmd_time = self.get_clock().now()

        self.timer = self.create_timer(self.update_period, self.update)
        self.tick = 0

        self.get_logger().info("Fake base started. Subscribing /cmd_vel.")

    def on_cmd_vel(self, msg):
        self.v = msg.linear.x
        self.w = msg.angular.z
        self.last_cmd_time = self.get_clock().now()

    def normalize_angle(self, a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    def quat_from_yaw(self, yaw):
        return math.sin(yaw * 0.5), math.cos(yaw * 0.5)

    def clamp(self, value, lo, hi):
        return max(lo, min(hi, value))

    def clamp_to_map_bounds(self, x, y):
        if not self.enforce_map_bounds:
            return x, y, False

        min_x = self.map_min_x + self.boundary_margin
        max_x = self.map_max_x - self.boundary_margin
        min_y = self.map_min_y + self.boundary_margin
        max_y = self.map_max_y - self.boundary_margin

        clamped_x = self.clamp(x, min_x, max_x)
        clamped_y = self.clamp(y, min_y, max_y)
        clamped = abs(clamped_x - x) > 1e-9 or abs(clamped_y - y) > 1e-9
        return clamped_x, clamped_y, clamped

    def update(self):
        dt = self.update_period

        now = self.get_clock().now()
        age = (now - self.last_cmd_time).nanoseconds * 1e-9

        if age > self.cmd_timeout:
            v = 0.0
            w = 0.0
        else:
            v = self.v
            w = self.w

        next_x = self.x + v * math.cos(self.yaw) * dt
        next_y = self.y + v * math.sin(self.yaw) * dt
        next_x, next_y, clamped = self.clamp_to_map_bounds(next_x, next_y)

        if clamped:
            v = 0.0
            self.v = 0.0

        self.x = next_x
        self.y = next_y
        self.yaw = self.normalize_angle(self.yaw + w * dt)

        self.publish_state()

        self.tick += 1
        if self.tick % 10 == 0:
            self.publish_start_pose()

    def publish_state(self):
        now = self.get_clock().now().to_msg()
        qz, qw = self.quat_from_yaw(self.yaw)

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = self.v
        odom.twist.twist.angular.z = self.w
        self.odom_pub.publish(odom)

        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = "map"
        pose.pose.position.x = self.x
        pose.pose.position.y = self.y
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        self.pose_pub.publish(pose)

        tf = TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf)

    def publish_start_pose(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = self.x
        msg.pose.position.y = self.y
        msg.pose.orientation.w = 1.0
        self.start_pub.publish(msg)


def main():
    rclpy.init()
    node = FakeBase()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()