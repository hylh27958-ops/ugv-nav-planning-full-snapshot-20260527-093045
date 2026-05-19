import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from tf2_ros import TransformBroadcaster


class FakeRobot(Node):
    def __init__(self):
        super().__init__("fake_robot")

        self.create_subscription(Path, "/astar_path", self.on_path, 10)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "/robot_pose", 10)
        self.start_pub = self.create_publisher(PoseStamped, "/start_pose", 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.x = 0.5
        self.y = 0.5
        self.yaw = 0.0
        self.speed = 0.35
        self.path = []
        self.target_index = 1

        self.timer = self.create_timer(0.05, self.update)
        self.get_logger().info("Fake robot started.")

    def on_path(self, msg):
        points = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if len(points) >= 2:
            self.path = points
            self.target_index = 1

    def quat_from_yaw(self, yaw):
        return math.sin(yaw * 0.5), math.cos(yaw * 0.5)

    def update(self):
        dt = 0.05

        if self.path and self.target_index < len(self.path):
            tx, ty = self.path[self.target_index]
            dx = tx - self.x
            dy = ty - self.y
            dist = math.hypot(dx, dy)

            if dist < 0.12:
                self.target_index += 1
            elif dist > 1e-6:
                self.yaw = math.atan2(dy, dx)
                step = min(self.speed * dt, dist)
                self.x += step * dx / dist
                self.y += step * dy / dist

        self.publish_state()

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

        start = PoseStamped()
        start.header.stamp = now
        start.header.frame_id = "map"
        start.pose.position.x = self.x
        start.pose.position.y = self.y
        start.pose.orientation.w = 1.0
        self.start_pub.publish(start)

        tf = TransformStamped()
        tf.header.stamp = now
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_link"
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tf)


def main():
    rclpy.init()
    node = FakeRobot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()