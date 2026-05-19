import heapq
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped


class AStarInteractive(Node):
    def __init__(self):
        super().__init__("astar_interactive")

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map_pub = self.create_publisher(OccupancyGrid, "/map", map_qos)
        self.path_pub = self.create_publisher(Path, "/astar_path", 10)
        self.raw_path_pub = self.create_publisher(Path, "/astar_path_raw", 10)
        self.start_pub = self.create_publisher(PoseStamped, "/astar_start", 10)
        self.goal_pub = self.create_publisher(PoseStamped, "/astar_goal", 10)

        self.create_subscription(PoseStamped, "/goal_pose", self.on_goal_pose, 10)
        self.create_subscription(PoseStamped, "/move_base_simple/goal", self.on_goal_pose, 10)
        self.create_subscription(PoseStamped, "/start_pose", self.on_start_pose, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.on_initial_pose, 10)

        self.width = 80
        self.height = 60
        self.resolution = 0.15
        self.origin_x = -1.0
        self.origin_y = -1.0
        self.inflate_radius = 0.25

        self.raw_grid = [0 for _ in range(self.width * self.height)]
        self.grid = [0 for _ in range(self.width * self.height)]

        self.start_world = (0.5, 0.5)
        self.goal_world = (9.5, 6.5)

        self.build_demo_map()
        self.inflate_obstacles()

        self.timer = self.create_timer(1.0, self.publish_all)
        self.get_logger().info("Interactive A* node started. Publish /goal_pose to replan.")

    def index(self, x, y):
        return y * self.width + x

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def set_cell(self, x, y, value):
        if self.in_bounds(x, y):
            self.raw_grid[self.index(x, y)] = value

    def is_blocked(self, x, y):
        if not self.in_bounds(x, y):
            return True
        return self.grid[self.index(x, y)] >= 50

    def world_to_grid(self, wx, wy):
        gx = int((wx - self.origin_x) / self.resolution)
        gy = int((wy - self.origin_y) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx, gy):
        wx = self.origin_x + (gx + 0.5) * self.resolution
        wy = self.origin_y + (gy + 0.5) * self.resolution
        return wx, wy

    def set_obstacle_rect(self, x0, y0, x1, y1):
        gx0, gy0 = self.world_to_grid(x0, y0)
        gx1, gy1 = self.world_to_grid(x1, y1)

        min_x, max_x = sorted((gx0, gx1))
        min_y, max_y = sorted((gy0, gy1))

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                self.set_cell(x, y, 100)

    def build_demo_map(self):
        # 竖向障碍，中间留一个通道
        self.set_obstacle_rect(3.6, 0.2, 3.9, 2.8)
        self.set_obstacle_rect(3.6, 3.7, 3.9, 7.0)

        # 横向障碍，中间留一个通道
        self.set_obstacle_rect(4.0, 4.1, 6.1, 4.4)
        self.set_obstacle_rect(7.1, 4.1, 9.2, 4.4)

        # 右下角障碍
        self.set_obstacle_rect(6.2, 0.0, 6.5, 2.6)

    def inflate_obstacles(self):
        self.grid = self.raw_grid.copy()
        radius_cells = max(1, int(math.ceil(self.inflate_radius / self.resolution)))

        occupied = []
        for y in range(self.height):
            for x in range(self.width):
                if self.raw_grid[self.index(x, y)] >= 100:
                    occupied.append((x, y))

        for ox, oy in occupied:
            for dy in range(-radius_cells, radius_cells + 1):
                for dx in range(-radius_cells, radius_cells + 1):
                    nx = ox + dx
                    ny = oy + dy
                    if not self.in_bounds(nx, ny):
                        continue
                    if math.hypot(dx, dy) <= radius_cells:
                        idx = self.index(nx, ny)
                        if self.grid[idx] == 0:
                            self.grid[idx] = 55

        for ox, oy in occupied:
            self.grid[self.index(ox, oy)] = 100

    def nearest_free(self, cell):
        if self.in_bounds(*cell) and not self.is_blocked(*cell):
            return cell

        cx, cy = cell
        for r in range(1, 20):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx = cx + dx
                    ny = cy + dy
                    if self.in_bounds(nx, ny) and not self.is_blocked(nx, ny):
                        return nx, ny
        return None

    def heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def astar(self, start, goal):
        start = self.nearest_free(start)
        goal = self.nearest_free(goal)

        if start is None or goal is None:
            return []

        open_set = []
        heapq.heappush(open_set, (0.0, start))

        came_from = {}
        g_score = {start: 0.0}
        closed = set()

        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current in closed:
                continue

            if current == goal:
                return self.reconstruct_path(came_from, current)

            closed.add(current)
            cx, cy = current

            for dx, dy in neighbors:
                nx = cx + dx
                ny = cy + dy

                if self.is_blocked(nx, ny):
                    continue

                if dx != 0 and dy != 0:
                    if self.is_blocked(cx + dx, cy) or self.is_blocked(cx, cy + dy):
                        continue

                neighbor = (nx, ny)
                step_cost = math.sqrt(2.0) if dx != 0 and dy != 0 else 1.0
                tentative_g = g_score[current] + step_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        return []

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def bresenham(self, a, b):
        x0, y0 = a
        x1, y1 = b

        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return cells

    def line_is_free(self, a, b):
        for x, y in self.bresenham(a, b):
            if self.is_blocked(x, y):
                return False
        return True

    def simplify_path(self, cells):
        if len(cells) <= 2:
            return cells

        result = [cells[0]]
        anchor = 0

        while anchor < len(cells) - 1:
            target = len(cells) - 1
            while target > anchor + 1:
                if self.line_is_free(cells[anchor], cells[target]):
                    break
                target -= 1

            result.append(cells[target])
            anchor = target

        return result

    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0

        msg.data = self.grid
        self.map_pub.publish(msg)

    def make_path_msg(self, cells):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        for gx, gy in cells:
            wx, wy = self.grid_to_world(gx, gy)

            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        return path_msg

    def publish_pose(self, publisher, world_xy):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = world_xy[0]
        msg.pose.position.y = world_xy[1]
        msg.pose.orientation.w = 1.0
        publisher.publish(msg)

    def publish_paths(self):
        start = self.world_to_grid(*self.start_world)
        goal = self.world_to_grid(*self.goal_world)

        raw_cells = self.astar(start, goal)
        simple_cells = self.simplify_path(raw_cells)

        self.raw_path_pub.publish(self.make_path_msg(raw_cells))
        self.path_pub.publish(self.make_path_msg(simple_cells))

        if raw_cells:
            self.get_logger().info(
                f"Path planned: raw={len(raw_cells)} points, simplified={len(simple_cells)} points"
            )
        else:
            self.get_logger().warn("A* failed: no valid path.")

    def publish_all(self):
        self.publish_map()
        self.publish_pose(self.start_pub, self.start_world)
        self.publish_pose(self.goal_pub, self.goal_world)
        self.publish_paths()

    def on_goal_pose(self, msg):
        self.goal_world = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(
            f"New goal: x={self.goal_world[0]:.2f}, y={self.goal_world[1]:.2f}"
        )
        self.publish_all()

    def on_start_pose(self, msg):
        self.start_world = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(
            f"New start: x={self.start_world[0]:.2f}, y={self.start_world[1]:.2f}"
        )
        self.publish_all()

    def on_initial_pose(self, msg):
        self.start_world = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.get_logger().info(
            f"New initial pose: x={self.start_world[0]:.2f}, y={self.start_world[1]:.2f}"
        )
        self.publish_all()


def main():
    rclpy.init()
    node = AStarInteractive()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()