import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from tf2_ros import TransformBroadcaster


class FakePathFollower(Node):
    def __init__(self):
        super().__init__("fake_path_follower")

        self.path_sub = self.create_subscription(Path, "/astar_path", self.on_path, 10)

        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "/robot_pose", 10)
        self.start_pub = self.create_publisher(PoseStamped, "/start_pose", 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.x = 0.5
        self.y = 0.5
        self.yaw = 0.0

        self.speed = 0.45
        self.waypoint_tolerance = 0.12
        self.path = []
        self.target_index = 0

        self.tick_count = 0
        self.timer = self.create_timer(0.05, self.on_timer)

        self.get_logger().info("Fake path follower started.")

    def on_path(self, msg):
        points = []
        for pose in msg.poses:
            points.append((pose.pose.position.x, pose.pose.position.y))

        if len(points) >= 2:
            self.path = points
            self.target_index = 1

    def yaw_to_quat(self, yaw):
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)
        return 0.0, 0.0, qz, qw

    def on_timer(self):
        dt = 0.05

        if self.path and self.target_index < len(self.path):
            tx, ty = self.path[self.target_index]
            dx = tx - self.x
            dy = ty - self.y
            dist = math.hypot(dx, dy)

            if dist < self.waypoint_tolerance:
                self.target_index += 1
            elif dist > 1e-6:
                self.yaw = math.atan2(dy, dx)
                step = min(self.speed * dt, dist)
                self.x += step * dx / dist
                self.y += step * dy / dist

        self.publish_state()

        self.tick_count += 1
        if self.tick_count % 10 == 0:
            self.publish_start_pose()

    def publish_state(self):
        now = self.get_clock().now().to_msg()
        qx, qy, qz, qw = self.yaw_to_quat(self.yaw)

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = self.speed
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
    node = FakePathFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()