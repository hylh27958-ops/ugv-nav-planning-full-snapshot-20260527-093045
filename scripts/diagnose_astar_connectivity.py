import math
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path


INFLATE_RADIUS_M = 0.18
OCCUPIED_THRESHOLD = 50


class AstarConnectivityDiagnostic(Node):
    def __init__(self):
        super().__init__("astar_connectivity_diagnostic")

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

    def idx(self, gx, gy):
        return gy * self.map_msg.info.width + gx

    def raw_value(self, gx, gy):
        if not self.in_bounds(gx, gy):
            return None
        return self.map_msg.data[self.idx(gx, gy)]

    def is_raw_occupied(self, gx, gy):
        value = self.raw_value(gx, gy)
        return value is None or value < 0 or value >= OCCUPIED_THRESHOLD

    def build_inflated_grid(self):
        info = self.map_msg.info
        width = info.width
        height = info.height
        resolution = info.resolution
        inflate_cells = int(math.ceil(INFLATE_RADIUS_M / resolution))

        blocked = [False] * (width * height)

        occupied = []
        for y in range(height):
            for x in range(width):
                if self.is_raw_occupied(x, y):
                    occupied.append((x, y))

        for ox, oy in occupied:
            for dy in range(-inflate_cells, inflate_cells + 1):
                for dx in range(-inflate_cells, inflate_cells + 1):
                    nx = ox + dx
                    ny = oy + dy
                    if not self.in_bounds(nx, ny):
                        continue
                    dist = math.hypot(dx, dy) * resolution
                    if dist <= INFLATE_RADIUS_M + 1e-9:
                        blocked[self.idx(nx, ny)] = True

        return blocked

    def is_free_in_inflated(self, blocked, gx, gy):
        if not self.in_bounds(gx, gy):
            return False
        return not blocked[self.idx(gx, gy)]

    def nearest_free_in_inflated(self, blocked, gx, gy, max_radius=50):
        if self.is_free_in_inflated(blocked, gx, gy):
            return gx, gy, 0.0

        resolution = self.map_msg.info.resolution
        best = None

        for radius in range(1, max_radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    nx = gx + dx
                    ny = gy + dy
                    if not self.is_free_in_inflated(blocked, nx, ny):
                        continue
                    dist = math.hypot(dx, dy) * resolution
                    if best is None or dist < best[2]:
                        best = (nx, ny, dist)
            if best is not None:
                return best

        return None

    def neighbor_report(self, blocked, gx, gy):
        rows = []
        for dy in [-1, 0, 1]:
            row = []
            for dx in [-1, 0, 1]:
                nx = gx + dx
                ny = gy + dy
                if not self.in_bounds(nx, ny):
                    row.append("OUT")
                    continue
                raw = self.raw_value(nx, ny)
                inflated = "B" if blocked[self.idx(nx, ny)] else "F"
                row.append(f"{raw}/{inflated}")
            rows.append(row)
        return rows

    def bfs_connected(self, blocked, start, goal, max_expand=20000):
        if start is None or goal is None:
            return False, 0, None

        sx, sy, _ = start
        gx, gy, _ = goal

        if not self.is_free_in_inflated(blocked, sx, sy):
            return False, 0, None

        if not self.is_free_in_inflated(blocked, gx, gy):
            return False, 0, None

        width = self.map_msg.info.width
        height = self.map_msg.info.height
        resolution = self.map_msg.info.resolution

        visited = set()
        parent = {}
        q = deque()
        q.append((sx, sy))
        visited.add((sx, sy))

        neighbors = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]

        expanded = 0

        while q and expanded < max_expand:
            x, y = q.popleft()
            expanded += 1

            if x == gx and y == gy:
                length = 0.0
                cur = (x, y)
                while cur in parent:
                    px, py = parent[cur]
                    length += math.hypot(cur[0] - px, cur[1] - py) * resolution
                    cur = (px, py)
                return True, expanded, length

            for dx, dy in neighbors:
                nx = x + dx
                ny = y + dy

                if not (0 <= nx < width and 0 <= ny < height):
                    continue

                if (nx, ny) in visited:
                    continue

                if blocked[self.idx(nx, ny)]:
                    continue

                visited.add((nx, ny))
                parent[(nx, ny)] = (x, y)
                q.append((nx, ny))

        return False, expanded, None

    def print_pose(self, label, pose_msg, blocked):
        if pose_msg is None:
            print(f"{label}: missing")
            return None

        x = pose_msg.pose.position.x
        y = pose_msg.pose.position.y
        gx, gy = self.world_to_grid(x, y)

        raw = self.raw_value(gx, gy)
        inflated_free = self.is_free_in_inflated(blocked, gx, gy)
        nearest = self.nearest_free_in_inflated(blocked, gx, gy)

        print(f"{label}: world=({x:.3f}, {y:.3f}) grid=({gx}, {gy})")
        print(f"{label}: raw_cell_value={raw}")
        print(f"{label}: inflated_free={inflated_free}")
        print(f"{label}: nearest_inflated_free={nearest}")
        print(f"{label}: 3x3 raw/inflated neighbors:")
        for row in self.neighbor_report(blocked, gx, gy):
            print("  " + " | ".join(row))

        return nearest

    def report(self):
        print("========== A* Inflated Connectivity Diagnostic ==========")

        if self.map_msg is None:
            print("map: missing")
            return

        info = self.map_msg.info
        print(f"map: width={info.width} height={info.height} resolution={info.resolution}")
        print(f"inflate_radius_m={INFLATE_RADIUS_M}")
        print(f"astar_path_poses={len(self.astar_path.poses) if self.astar_path else 'missing'}")

        blocked = self.build_inflated_grid()

        start = self.print_pose("robot_pose", self.robot_pose, blocked)
        goal = self.print_pose("astar_goal", self.goal_pose, blocked)

        connected, expanded, approx_len = self.bfs_connected(blocked, start, goal)

        print(f"inflated_connected={connected}")
        print(f"bfs_expanded_cells={expanded}")
        print(f"approx_path_length_m={approx_len}")

        if not connected:
            print("DIAGNOSIS: start/goal are not connected in the inflated grid.")
        else:
            print("DIAGNOSIS: inflated grid is connected; A* implementation or state update likely needs inspection.")

        print("=========================================================")


def main():
    rclpy.init()
    node = AstarConnectivityDiagnostic()

    deadline = time.time() + 5.0
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if (
            node.map_msg is not None
            and node.robot_pose is not None
            and node.goal_pose is not None
            and node.astar_path is not None
        ):
            break

    node.report()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
