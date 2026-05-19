import math
import time
from numbers import Number

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
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
        self.declare_parameter("resend_period", 1.0)
        self.declare_parameter("feedback_timeout", 2.0)
        self.declare_parameter("goal_tolerance", 0.35)
        self.declare_parameter("goal_change_distance", 0.25)
        self.declare_parameter("min_path_poses", 2)
        self.declare_parameter("periodic_replan", True)

        self.action_name = self.get_parameter("action_name").value
        self.controller_id = self.get_parameter("controller_id").value
        self.goal_checker_id = self.get_parameter("goal_checker_id").value
        self.update_period = float(self.get_parameter("update_period").value)
        self.resend_period = float(self.get_parameter("resend_period").value)
        self.feedback_timeout = float(self.get_parameter("feedback_timeout").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.goal_change_distance = float(self.get_parameter("goal_change_distance").value)
        self.min_path_poses = int(self.get_parameter("min_path_poses").value)
        self.periodic_replan = bool(self.get_parameter("periodic_replan").value)

        action_topic = self.action_name
        if not action_topic.startswith("/"):
            action_topic = "/" + action_topic

        self.action_client = ActionClient(self, FollowPath, action_topic)

        self.create_subscription(Path, "/astar_path", self.on_path, 10)
        self.create_subscription(PoseStamped, "/robot_pose", self.on_robot_pose, 10)

        self.status_pub = self.create_publisher(String, "/teb_path_sender/status", 10)
        self.feedback_pub = self.create_publisher(String, "/teb_path_sender/feedback", 10)

        self.latest_path = None
        self.latest_goal_xy = None
        self.robot_pose_xy = None

        self.goal_handle = None
        self.goal_active = False
        self.goal_send_time = 0.0
        self.last_send_time = 0.0
        self.last_feedback_time = 0.0
        self.last_result_status = None

        self.timer = self.create_timer(self.update_period, self.on_timer)

        self.publish_status("WAITING_FOR_PATH")
        self.get_logger().info(
            f"Teb path sender started. action={action_topic}, controller={self.controller_id}"
        )

    def now_seconds(self):
        return time.time()

    def publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def publish_feedback(self, text):
        msg = String()
        msg.data = text
        self.feedback_pub.publish(msg)

    def on_robot_pose(self, msg):
        self.robot_pose_xy = (msg.pose.position.x, msg.pose.position.y)

    def on_path(self, msg):
        if len(msg.poses) < self.min_path_poses:
            return

        cleaned = self.clean_path(msg)
        if len(cleaned.poses) < self.min_path_poses:
            return

        self.latest_path = cleaned

        last_pose = cleaned.poses[-1].pose.position
        new_goal_xy = (last_pose.x, last_pose.y)

        if self.latest_goal_xy is None:
            self.latest_goal_xy = new_goal_xy
            self.goal_active = False
            return

        if self.distance_xy(self.latest_goal_xy, new_goal_xy) > self.goal_change_distance:
            self.latest_goal_xy = new_goal_xy
            self.goal_active = False
            self.publish_status("NEW_GOAL_RECEIVED")

    def clean_path(self, msg):
        output = Path()
        output.header = msg.header

        last_xy = None
        min_separation = 0.03

        for pose in msg.poses:
            xy = (pose.pose.position.x, pose.pose.position.y)
            if last_xy is None or self.distance_xy(last_xy, xy) >= min_separation:
                output.poses.append(pose)
                last_xy = xy

        if msg.poses:
            final_pose = msg.poses[-1]
            if not output.poses:
                output.poses.append(final_pose)
            else:
                output_goal = output.poses[-1].pose.position
                final_goal = final_pose.pose.position
                if math.hypot(output_goal.x - final_goal.x, output_goal.y - final_goal.y) > 1e-6:
                    output.poses.append(final_pose)

        return output

    def distance_xy(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def distance_to_goal(self):
        if self.robot_pose_xy is None or self.latest_goal_xy is None:
            return None
        return self.distance_xy(self.robot_pose_xy, self.latest_goal_xy)

    def should_send_goal(self):
        now = self.now_seconds()

        if self.latest_path is None:
            self.publish_status("WAITING_FOR_PATH")
            return False

        if len(self.latest_path.poses) < self.min_path_poses:
            self.publish_status("PATH_TOO_SHORT")
            return False

        if not self.action_client.wait_for_server(timeout_sec=0.0):
            self.publish_status("WAITING_FOR_FOLLOW_PATH_SERVER")
            return False

        dist = self.distance_to_goal()
        if dist is not None and dist <= self.goal_tolerance:
            self.goal_active = False
            self.publish_status("REACHED")
            self.publish_feedback(f"distance_to_goal={dist:.3f}")
            return False

        if not self.goal_active:
            return now - self.last_send_time >= self.resend_period

        feedback_age = now - self.last_feedback_time
        if self.last_feedback_time > 0.0 and feedback_age > self.feedback_timeout:
            self.goal_active = False
            self.publish_status(f"FEEDBACK_TIMEOUT_{feedback_age:.2f}s")
            return now - self.last_send_time >= self.resend_period

        if self.periodic_replan and now - self.goal_send_time > self.resend_period:
            self.goal_active = False
            self.publish_status("PERIODIC_REPLAN")
            return True

        self.publish_status("ACTIVE")
        return False

    def on_timer(self):
        if self.should_send_goal():
            self.send_goal()

    def send_goal(self):
        goal_msg = FollowPath.Goal()
        goal_msg.path = self.latest_path
        goal_msg.controller_id = self.controller_id
        goal_msg.goal_checker_id = self.goal_checker_id

        self.goal_active = True
        self.goal_send_time = self.now_seconds()
        self.last_send_time = self.goal_send_time
        self.last_feedback_time = self.goal_send_time

        future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.on_feedback,
        )
        future.add_done_callback(self.on_goal_response)

        self.publish_status(f"SENDING_GOAL poses={len(goal_msg.path.poses)}")

    def on_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.goal_handle = None
            self.goal_active = False
            self.publish_status("GOAL_RESPONSE_ERROR")
            self.get_logger().error(f"Goal response failed: {exc}")
            return

        if not goal_handle.accepted:
            self.goal_handle = None
            self.goal_active = False
            self.publish_status("GOAL_REJECTED")
            return

        self.goal_handle = goal_handle
        self.goal_active = True
        self.last_feedback_time = self.now_seconds()
        self.publish_status("GOAL_ACCEPTED")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.on_result)

    def read_feedback_float(self, feedback, field_name, default):
        if not hasattr(feedback, field_name):
            return default

        value = getattr(feedback, field_name)

        if isinstance(value, Number):
            return float(value)

        if hasattr(value, "data") and isinstance(value.data, Number):
            return float(value.data)

        if hasattr(value, "twist"):
            twist = value.twist
            if hasattr(twist, "linear") and hasattr(twist.linear, "x"):
                return float(twist.linear.x)

        if hasattr(value, "linear") and hasattr(value.linear, "x"):
            return float(value.linear.x)

        return default

    def on_feedback(self, feedback_msg):
        self.last_feedback_time = self.now_seconds()

        feedback = feedback_msg.feedback
        speed = self.read_feedback_float(feedback, "speed", 0.0)
        distance = self.read_feedback_float(feedback, "distance_to_goal", -1.0)

        self.publish_feedback(f"distance_to_goal={distance:.3f}, speed={speed:.3f}")

    def on_result(self, future):
        try:
            result = future.result()
        except Exception as exc:
            self.goal_active = False
            self.goal_handle = None
            self.publish_status("RESULT_ERROR")
            self.get_logger().error(f"Goal result failed: {exc}")
            return

        self.last_result_status = result.status
        self.goal_active = False
        self.goal_handle = None

        status_name = self.status_name(result.status)
        self.publish_status(f"RESULT_{status_name}")

    def status_name(self, status):
        names = {
            GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
            GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
            GoalStatus.STATUS_EXECUTING: "EXECUTING",
            GoalStatus.STATUS_CANCELING: "CANCELING",
            GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
            GoalStatus.STATUS_CANCELED: "CANCELED",
            GoalStatus.STATUS_ABORTED: "ABORTED",
        }
        return names.get(status, str(status))


def main():
    rclpy.init()
    node = TebPathSender()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
