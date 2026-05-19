import math

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from nav2_msgs.action import FollowPath
from nav_msgs.msg import Path
from std_msgs.msg import String


class TebPathSender(Node):
    def __init__(self):
        super().__init__("teb_path_sender")

        self.declare_parameter("action_name", "follow_path")
        self.declare_parameter("controller_id", "FollowPath")
        self.declare_parameter("goal_checker_id", "goal_checker")
        self.declare_parameter("update_period", 0.2)
        self.declare_parameter("resend_period", 2.0)
        self.declare_parameter("periodic_replan", True)
        self.declare_parameter("goal_change_distance", 0.25)
        self.declare_parameter("min_path_poses", 2)

        self.action_name = str(self.get_parameter("action_name").value)
        self.controller_id = str(self.get_parameter("controller_id").value)
        self.goal_checker_id = str(self.get_parameter("goal_checker_id").value)
        self.update_period = float(self.get_parameter("update_period").value)
        self.resend_period = float(self.get_parameter("resend_period").value)
        self.periodic_replan = bool(self.get_parameter("periodic_replan").value)
        self.goal_change_distance = float(self.get_parameter("goal_change_distance").value)
        self.min_path_poses = int(self.get_parameter("min_path_poses").value)

        self.create_subscription(Path, "/astar_path", self.on_path, 10)

        self.status_pub = self.create_publisher(String, "/teb_path_sender/status", 10)
        self.feedback_pub = self.create_publisher(String, "/teb_path_sender/feedback", 10)

        self.client = ActionClient(self, FollowPath, self.action_name)

        self.latest_path = None
        self.last_sent_goal_xy = None
        self.last_send_time = None
        self.pending_send = False
        self.server_warned = False

        self.timer = self.create_timer(self.update_period, self.update)

        self.get_logger().info(
            f"TEB path sender started. Sending /astar_path to action '{self.action_name}'."
        )

    def on_path(self, msg):
        if len(msg.poses) >= self.min_path_poses:
            self.latest_path = msg

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def path_goal_xy(self, path):
        pose = path.poses[-1].pose.position
        return pose.x, pose.y

    def distance(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def publish_feedback(self, text):
        msg = String()
        msg.data = text
        self.feedback_pub.publish(msg)

    def should_send_path(self):
        if self.latest_path is None:
            return False

        if self.pending_send:
            return False

        goal_xy = self.path_goal_xy(self.latest_path)
        now = self.now_sec()

        if self.last_sent_goal_xy is None:
            return True

        if self.distance(goal_xy, self.last_sent_goal_xy) > self.goal_change_distance:
            return True

        if self.periodic_replan and self.last_send_time is not None:
            if now - self.last_send_time > self.resend_period:
                return True

        return False

    def update(self):
        if self.latest_path is None:
            self.publish_status("WAITING_FOR_ASTAR_PATH")
            return

        if not self.client.server_is_ready():
            if not self.server_warned:
                self.get_logger().warn(f"Waiting for FollowPath action server: {self.action_name}")
                self.server_warned = True
            self.publish_status("WAITING_FOR_FOLLOW_PATH_SERVER")
            return

        if not self.should_send_path():
            self.publish_status("IDLE")
            return

        self.send_path()

    def send_path(self):
        goal_msg = FollowPath.Goal()
        goal_msg.path = self.latest_path
        goal_msg.controller_id = self.controller_id
        goal_msg.goal_checker_id = self.goal_checker_id

        goal_xy = self.path_goal_xy(self.latest_path)
        self.pending_send = True

        self.get_logger().info(
            f"Sending path to TEB: poses={len(self.latest_path.poses)}, "
            f"goal=({goal_xy[0]:.2f}, {goal_xy[1]:.2f})"
        )
        self.publish_status("SENDING_PATH")

        send_future = self.client.send_goal_async(
            goal_msg,
            feedback_callback=self.on_feedback,
        )
        send_future.add_done_callback(self.on_goal_response)

    def on_goal_response(self, future):
        self.pending_send = False

        goal_handle = future.result()
        if goal_handle is None:
            self.get_logger().error("FollowPath goal response is None.")
            self.publish_status("GOAL_RESPONSE_NONE")
            return

        if not goal_handle.accepted:
            self.get_logger().warn("FollowPath goal was rejected.")
            self.publish_status("GOAL_REJECTED")
            return

        self.last_sent_goal_xy = self.path_goal_xy(self.latest_path)
        self.last_send_time = self.now_sec()

        self.get_logger().info("FollowPath goal accepted.")
        self.publish_status("GOAL_ACCEPTED")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.on_result)

    def on_feedback(self, feedback_msg):
        feedback = feedback_msg.feedback
        distance_to_goal = getattr(feedback, "distance_to_goal", None)
        speed = getattr(feedback, "speed", None)

        if distance_to_goal is not None and speed is not None:
            self.publish_feedback(
                f"distance_to_goal={distance_to_goal:.3f}, speed={speed:.3f}"
            )
        elif distance_to_goal is not None:
            self.publish_feedback(f"distance_to_goal={distance_to_goal:.3f}")
        else:
            self.publish_feedback("feedback_received")

    def on_result(self, future):
        result_wrapper = future.result()
        result = result_wrapper.result

        error_code = getattr(result, "error_code", 0)
        error_msg = getattr(result, "error_msg", "")

        if error_code == 0:
            self.get_logger().info("FollowPath finished successfully.")
            self.publish_status("FOLLOW_PATH_SUCCEEDED")
        else:
            self.get_logger().warn(
                f"FollowPath finished with error_code={error_code}, error_msg='{error_msg}'"
            )
            self.publish_status(f"FOLLOW_PATH_ERROR_{error_code}: {error_msg}")


def main():
    rclpy.init()
    node = TebPathSender()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()