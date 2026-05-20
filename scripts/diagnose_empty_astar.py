import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path


class EmptyAstarDiagnostic(Node):
    def __init__(self):
        super().__init__("empty_astar_diagnostic")

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map_msg = None
        self.robot_pose = None
        self.goal_pose = None
        self.astar_path = None

        self.create_subscription(OccupancyGrid, "/map", self.on_map, map_qos)
        self.create_subscription(PoseStamped, "/robot_pose", self.on_robot_pose, 10)
        self.create_subscription(PoseStamped, "/astar_goal", self.on_goal_pose, 10)
        self.create_subscription(Path, "/astar_path", self.on_astar_path, 10)

    def on_map(self, msg):
        self.map_msg = msg

    def on_robot_pose(self, msg):
        self.robot_pose = msg

    def on_goal_pose(self, msg):
        self.goal_pose = msg

    def on_astar_path(self, msg):
        self.astar_path = msg

    def world_to_grid(self, x, y):
        info = self.map_msg.info
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)
        return gx, gy

    def in_bounds(self, gx, gy):
        info = self.map_msg.info
        return 0 <= gx < info.width and 0 <= gy < info.height

    def cell_value(self, gx, gy):
        info = self.map_msg.info
        if not self.in_bounds(gx, gy):
            return None
        return self.map_msg.data[gy * info.width + gx]

    def nearest_free(self, gx, gy, max_radius=30):
        if not self.in_bounds(gx, gy):
            return None

        for radius in range(max_radius + 1):
            best = None
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    nx = gx + dx
                    ny = gy + dy
                    if not self.in_bounds(nx, ny):
                        continue
                    value = self.cell_value(nx, ny)
                    if value == 0:
                        dist = math.hypot(dx, dy) * self.map_msg.info.resolution
                        best = (nx, ny, dist)
                        return best
        return None

    def nearest_occupied_distance(self, gx, gy, max_radius=30):
        if not self.in_bounds(gx, gy):
            return None

        best = None
        for dy in range(-max_radius, max_radius + 1):
            for dx in range(-max_radius, max_radius + 1):
                nx = gx + dx
                ny = gy + dy
                if not self.in_bounds(nx, ny):
                    continue
                value = self.cell_value(nx, ny)
                if value is not None and value >= 50:
                    dist = math.hypot(dx, dy) * self.map_msg.info.resolution
                    if best is None or dist < best:
                        best = dist
        return best

    def print_pose_report(self, label, pose_msg):
        if pose_msg is None:
            print(f"{label}: missing")
            return

        x = pose_msg.pose.position.x
        y = pose_msg.pose.position.y
        gx, gy = self.world_to_grid(x, y)
        value = self.cell_value(gx, gy)
        nearest_free = self.nearest_free(gx, gy)
        nearest_occ = self.nearest_occupied_distance(gx, gy)

        print(f"{label}: world=({x:.3f}, {y:.3f}) grid=({gx}, {gy}) cell_value={value}")
        print(f"{label}: nearest_free={nearest_free}")
        print(f"{label}: nearest_occupied_distance_m={nearest_occ}")

    def report(self):
        print("========== Empty A* Diagnostic ==========")

        if self.map_msg is None:
            print("map: missing")
            return

        info = self.map_msg.info
        print(f"map: width={info.width} height={info.height} resolution={info.resolution}")
        print(f"map origin: ({info.origin.position.x}, {info.origin.position.y})")

        if self.astar_path is None:
            print("astar_path: missing")
        else:
            print(f"astar_path poses: {len(self.astar_path.poses)}")

        self.print_pose_report("robot_pose", self.robot_pose)
        self.print_pose_report("astar_goal", self.goal_pose)

        print("cell_value meaning: 0=free, 100=occupied, -1=unknown")
        print("=========================================")


def main():
    rclpy.init()
    node = EmptyAstarDiagnostic()

    deadline = time.time() + 5.0
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.map_msg is not None and node.robot_pose is not None and node.goal_pose is not None and node.astar_path is not None:
            break

    node.report()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
