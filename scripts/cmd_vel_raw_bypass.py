#!/usr/bin/env python3
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdVelRawBypass(Node):
    def __init__(self):
        super().__init__("cmd_vel_raw_bypass")
        self.declare_parameter("stale_timeout_s", 0.5)

        self.stale_timeout_s = float(self.get_parameter("stale_timeout_s").value)
        self.last_raw_time = None

        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Twist, "/cmd_vel_raw", self.on_raw, 10)
        self.create_timer(0.1, self.on_timer)

        self.get_logger().warn("Safety filter bypass active: forwarding /cmd_vel_raw to /cmd_vel")

    def on_raw(self, msg):
        self.last_raw_time = self.get_clock().now()
        self.pub.publish(msg)

    def on_timer(self):
        if self.last_raw_time is None:
            self.pub.publish(Twist())
            return

        age = (self.get_clock().now() - self.last_raw_time).nanoseconds / 1e9
        if age > self.stale_timeout_s:
            self.pub.publish(Twist())


def main():
    rclpy.init()
    node = CmdVelRawBypass()
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
