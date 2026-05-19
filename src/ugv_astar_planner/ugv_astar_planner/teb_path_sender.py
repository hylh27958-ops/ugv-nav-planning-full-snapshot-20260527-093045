import math
import time
from numbers import Number

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Quaternion
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
        self.declare_parameter("feedback_timeout", 4.0)
        self.declare_parameter("goal_tolerance", 0.35)
        self.declare_parameter("goal_change_distance", 0.25)
        self.declare_parameter("min_path_poses", 2)
        self.declare_parameter("periodic_replan", False)
        self.declare_parameter("waypoint_spacing", 0.15)
        self.declare_parameter("progress_timeout", 8.0)
        self.declare_parameter("progress_epsilon", 0.10)

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
        self.waypoint_spacing = float(self.get_parameter("waypoint_spacing").value)
        self.progress_timeout = float(self.get_parameter("progress_timeout").value)
        self.progress_epsilon = float(self.get_parameter("progress_epsilon").value)

        action_topic = self.action_name
        if not action_topic.startswith("/"):
            action_topic = "/" + action_topic

        self.action_client = ActionClient(self, FollowPath, action_topic)

        self.create_subscription(Path, "/astar_path", self.on_path, 10)
        self.create_subscription(PoseStamped, "/robot_pose", self.on_robot_pose, 10)

        self.status_pub = self.create_publisher(String, "/teb_path_sender/status", 10)
        self.feedback_pub = self.create_publisher(String, "/teb_path_sender/feedback", 10)

        self.latest_raw_path = None
        self.latest_goal_xy = None
        self.robot_pose = None
        self.robot_pose_xy = None

        self.goal_handle = None
        self.goal_active = False
        self.goal_send_time = 0.0
        self.last_send_time = 0.0
        self.last_feedback_time = 0.0
        self.last_result_status = None

        self.best_feedback_distance = None
        self.last_progress_time = 0.0

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

    def yaw_to_quaternion(self, yaw):
        q = Quaternion()
        q.z = math.sin(yaw * 0.5)
        q.w = math.cos(yaw * 0.5)
        return q

    def on_robot_pose(self, msg):
        self.robot_pose = msg
        self.robot_pose_xy = (msg.pose.position.x, msg.pose.position.y)

    def on_path(self, msg):
        if len(msg.poses) < self.min_path_poses:
            return

        self.latest_raw_path = msg

        last_pose = msg.poses[-1].pose.position
        new_goal_xy = (last_pose.x, last_pose.y)

        if self.latest_goal_xy is None:
            self.latest_goal_xy = new_goal_xy
            self.goal_active = False
            return

        if self.distance_xy(self.latest_goal_xy, new_goal_xy) > self.goal_change_distance:
            self.latest_goal_xy = new_goal_xy
            self.goal_active = False
            self.best_feedback_distance = None
            self.last_progress_time = self.now_seconds()
            self.publish_status("NEW_GOAL_RECEIVED")

    def distance_xy(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def distance_to_goal(self):
        if self.robot_pose_xy is None or self.latest_goal_xy is None:
            return None
        return self.distance_xy(self.robot_pose_xy, self.latest_goal_xy)

    def pose_xy(self, pose_stamped):
        p = pose_stamped.pose.position
        return (p.x, p.y)

    def prune_points_near_robot(self, points):
        if self.robot_pose_xy is None or len(points) < 2:
            return points

        nearest_idx = 0
        nearest_dist = None

        for i, point in enumerate(points):
            dist = self.distance_xy(self.robot_pose_xy, point)
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist
                nearest_idx = i

        start_idx = max(0, nearest_idx - 1)
        pruned = points[start_idx:]

        if not pruned:
            return points

        robot_to_first = self.distance_xy(self.robot_pose_xy, pruned[0])
        if robot_to_first > 0.05:
            pruned = [self.robot_pose_xy] + pruned

        return pruned

    def densify_points(self, points):
        if len(points) < 2:
            return points

        dense = [points[0]]

        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]
            dist = math.hypot(x1 - x0, y1 - y0)

            if dist < 1e-6:
                continue

            steps = max(1, int(math.ceil(dist / self.waypoint_spacing)))

            for step in range(1, steps + 1):
                t = step / steps
                dense.append((
                    x0 + (x1 - x0) * t,
                    y0 + (y1 - y0) * t,
                ))

        return dense

    def build_goal_path(self):
        if self.latest_raw_path is None:
            return None

        raw_points = [
            self.pose_xy(pose)
            for pose in self.latest_raw_path.poses
        ]

        if len(raw_points) < self.min_path_poses:
            return None

        pruned = self.prune_points_near_robot(raw_points)
        dense = self.densify_points(pruned)

        if len(dense) < self.min_path_poses:
            return None

        output = Path()
        output.header = self.latest_raw_path.header
        output.header.stamp = self.get_clock().now().to_msg()

        for i, (x, y) in enumerate(dense):
            pose = PoseStamped()
            pose.header = output.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            if i < len(dense) - 1:
                nx, ny = dense[i + 1]
                yaw = math.atan2(ny - y, nx - x)
            elif len(dense) >= 2:
                px, py = dense[i - 1]
                yaw = math.atan2(y - py, x - px)
            else:
                yaw = 0.0

            pose.pose.orientation = self.yaw_to_quaternion(yaw)
            output.poses.append(pose)

        return output

    def should_send_goal(self):
        now = self.now_seconds()

        if self.latest_raw_path is None:
            self.publish_status("WAITING_FOR_PATH")
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

        if self.last_progress_time > 0.0 and now - self.last_progress_time > self.progress_timeout:
            self.goal_active = False
            self.publish_status("NO_PROGRESS_REPLAN")
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
        goal_path = self.build_goal_path()
        if goal_path is None or len(goal_path.poses) < self.min_path_poses:
            self.publish_status("PATH_TOO_SHORT")
            return

        goal_msg = FollowPath.Goal()
        goal_msg.path = goal_path
        goal_msg.controller_id = self.controller_id
        goal_msg.goal_checker_id = self.goal_checker_id

        self.goal_active = True
        self.goal_send_time = self.now_seconds()
        self.last_send_time = self.goal_send_time
        self.last_feedback_time = self.goal_send_time
        self.last_progress_time = self.goal_send_time
        self.best_feedback_distance = None

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
        self.last_progress_time = self.now_seconds()
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

        if distance >= 0.0:
            if self.best_feedback_distance is None:
                self.best_feedback_distance = distance
                self.last_progress_time = self.now_seconds()
            elif distance < self.best_feedback_distance - self.progress_epsilon:
                self.best_feedback_distance = distance
                self.last_progress_time = self.now_seconds()

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
