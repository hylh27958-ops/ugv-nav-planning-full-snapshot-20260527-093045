import json
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray


class LocalTrajectoryPlanner(Node):
    def __init__(self):
        super().__init__("local_trajectory_planner")

        self.declare_parameter("update_period", 0.1)
        self.declare_parameter("horizon_distance", 3.0)
        self.declare_parameter("sample_step", 0.15)
        self.declare_parameter("obstacle_influence", 0.9)
        self.declare_parameter("max_obstacle_shift", 0.35)
        self.declare_parameter("smooth_iterations", 3)

        self.update_period = float(self.get_parameter("update_period").value)
        self.horizon_distance = float(self.get_parameter("horizon_distance").value)
        self.sample_step = float(self.get_parameter("sample_step").value)
        self.obstacle_influence = float(self.get_parameter("obstacle_influence").value)
        self.max_obstacle_shift = float(self.get_parameter("max_obstacle_shift").value)
        self.smooth_iterations = int(self.get_parameter("smooth_iterations").value)

        self.horizon_scale = 1.0
        self.obstacle_influence_scale = 1.0
        self.obstacle_shift_scale = 1.0

        self.create_subscription(Path, "/astar_path", self.on_global_path, 10)
        self.create_subscription(PoseStamped, "/robot_pose", self.on_robot_pose, 10)
        self.create_subscription(MarkerArray, "/dynamic_obstacles", self.on_dynamic_obstacles, 10)
        self.create_subscription(String, "/sac/adaptation", self.on_adaptation, 10)

        self.local_path_pub = self.create_publisher(Path, "/local_trajectory", 10)
        self.local_goal_pub = self.create_publisher(PoseStamped, "/local_goal", 10)

        self.global_path = []
        self.dynamic_obstacles = []
        self.has_pose = False
        self.x = 0.0
        self.y = 0.0

        self.timer = self.create_timer(self.update_period, self.update)
        self.get_logger().info("Local trajectory planner started.")

    def on_adaptation(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        self.horizon_scale = float(data.get("horizon_scale", 1.0))
        self.obstacle_influence_scale = float(data.get("obstacle_influence_scale", 1.0))
        self.obstacle_shift_scale = float(data.get("obstacle_shift_scale", 1.0))

    def on_global_path(self, msg):
        self.global_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def on_robot_pose(self, msg):
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.has_pose = True

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

    def scaled_horizon_distance(self):
        return max(0.8, self.horizon_distance * self.horizon_scale)

    def scaled_obstacle_influence(self):
        return max(0.2, self.obstacle_influence * self.obstacle_influence_scale)

    def scaled_max_obstacle_shift(self):
        return max(0.05, self.max_obstacle_shift * self.obstacle_shift_scale)

    def distance_to_robot(self, point):
        return math.hypot(point[0] - self.x, point[1] - self.y)

    def distance(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def nearest_global_index(self):
        return min(
            range(len(self.global_path)),
            key=lambda i: self.distance_to_robot(self.global_path[i]),
        )

    def extract_local_segment(self):
        if len(self.global_path) < 2:
            return []

        nearest_i = self.nearest_global_index()
        segment = [(self.x, self.y)]
        total = 0.0
        last = self.global_path[nearest_i]
        horizon = self.scaled_horizon_distance()

        for i in range(nearest_i, len(self.global_path)):
            point = self.global_path[i]
            total += self.distance(last, point)
            segment.append(point)
            last = point

            if total >= horizon:
                break

        if len(segment) < 2:
            segment.append(self.global_path[-1])

        return segment

    def densify_path(self, points):
        if len(points) < 2:
            return points

        dense = [points[0]]

        for i in range(len(points) - 1):
            a = points[i]
            b = points[i + 1]
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            length = math.hypot(dx, dy)

            if length < 1e-6:
                continue

            steps = max(1, int(length / self.sample_step))

            for s in range(1, steps + 1):
                t = s / steps
                dense.append((a[0] + t * dx, a[1] + t * dy))

        return dense

    def apply_dynamic_obstacle_repulsion(self, points):
        if len(points) <= 2:
            return points

        adjusted = [points[0]]
        influence_base = self.scaled_obstacle_influence()
        max_shift = self.scaled_max_obstacle_shift()

        for i in range(1, len(points) - 1):
            px, py = points[i]
            shift_x = 0.0
            shift_y = 0.0

            for ox, oy, radius in self.dynamic_obstacles:
                dx = px - ox
                dy = py - oy
                dist = math.hypot(dx, dy)
                influence = influence_base + radius

                if dist < 1e-6 or dist > influence:
                    continue

                strength = (influence - dist) / influence
                shift_x += strength * dx / dist * max_shift
                shift_y += strength * dy / dist * max_shift

            shift_len = math.hypot(shift_x, shift_y)
            if shift_len > max_shift:
                scale = max_shift / shift_len
                shift_x *= scale
                shift_y *= scale

            adjusted.append((px + shift_x, py + shift_y))

        adjusted.append(points[-1])
        return adjusted

    def smooth_path(self, points):
        if len(points) <= 2:
            return points

        smoothed = points[:]

        for _ in range(self.smooth_iterations):
            new_points = [smoothed[0]]
            for i in range(1, len(smoothed) - 1):
                prev_p = smoothed[i - 1]
                cur_p = smoothed[i]
                next_p = smoothed[i + 1]
                x = 0.25 * prev_p[0] + 0.5 * cur_p[0] + 0.25 * next_p[0]
                y = 0.25 * prev_p[1] + 0.5 * cur_p[1] + 0.25 * next_p[1]
                new_points.append((x, y))
            new_points.append(smoothed[-1])
            smoothed = new_points

        return smoothed

    def make_path_msg(self, points):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        for x, y in points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        return msg

    def publish_local_goal(self, point):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = point[0]
        msg.pose.position.y = point[1]
        msg.pose.orientation.w = 1.0
        self.local_goal_pub.publish(msg)

    def update(self):
        if not self.has_pose or len(self.global_path) < 2:
            self.local_path_pub.publish(self.make_path_msg([]))
            return

        segment = self.extract_local_segment()
        dense = self.densify_path(segment)
        repulsed = self.apply_dynamic_obstacle_repulsion(dense)
        local = self.smooth_path(repulsed)

        self.local_path_pub.publish(self.make_path_msg(local))

        if local:
            self.publish_local_goal(local[-1])


def main():
    rclpy.init()
    node = LocalTrajectoryPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()