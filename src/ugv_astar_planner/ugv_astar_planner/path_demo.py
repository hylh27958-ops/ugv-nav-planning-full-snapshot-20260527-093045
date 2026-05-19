import heapq
import math
import random

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray


class AStarPlannerDemo(Node):
    def __init__(self):
        super().__init__("astar_planner_demo")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("width", 140),
                ("height", 90),
                ("resolution", 0.1),
                ("origin_x", 0.0),
                ("origin_y", 0.0),
                ("inflate_radius", 0.18),
                ("start_x", 0.8),
                ("start_y", 0.8),
                ("goal_x", 12.8),
                ("goal_y", 8.0),
                ("timer_period", 0.2),
                ("random_seed", 8),
                ("random_obstacle_count", 120),
                ("block_min_w", 2),
                ("block_max_w", 10),
                ("block_min_h", 2),
                ("block_max_h", 8),
                ("route_corridor_cells", 4),
                ("dynamic_obstacle_radius", 0.35),
                (
                    "dynamic_obstacles",
                    "4.6,1.6,0.20,0.00,3.5,6.0,1.6,1.6;"
                    "6.5,4.3,0.00,0.18,6.5,6.5,3.0,5.6;"
                    "9.2,6.7,-0.16,0.00,8.0,10.5,6.7,6.7;"
                    "10.7,7.5,0.00,-0.14,10.7,10.7,6.0,8.2",
                ),
            ],
        )

        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.resolution = float(self.get_parameter("resolution").value)
        self.origin_x = float(self.get_parameter("origin_x").value)
        self.origin_y = float(self.get_parameter("origin_y").value)
        self.inflate_radius = float(self.get_parameter("inflate_radius").value)

        self.start_world = (
            float(self.get_parameter("start_x").value),
            float(self.get_parameter("start_y").value),
        )
        self.goal_world = (
            float(self.get_parameter("goal_x").value),
            float(self.get_parameter("goal_y").value),
        )

        self.timer_period = float(self.get_parameter("timer_period").value)
        self.random_seed = int(self.get_parameter("random_seed").value)
        self.random_obstacle_count = int(self.get_parameter("random_obstacle_count").value)
        self.block_min_w = int(self.get_parameter("block_min_w").value)
        self.block_max_w = int(self.get_parameter("block_max_w").value)
        self.block_min_h = int(self.get_parameter("block_min_h").value)
        self.block_max_h = int(self.get_parameter("block_max_h").value)
        self.route_corridor_cells = int(self.get_parameter("route_corridor_cells").value)
        self.dynamic_obstacle_radius = float(self.get_parameter("dynamic_obstacle_radius").value)
        self.dynamic_obstacles = self.parse_dynamic_obstacles(
            self.get_parameter("dynamic_obstacles").value
        )

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
        self.dynamic_pub = self.create_publisher(MarkerArray, "/dynamic_obstacles", 10)

        self.create_subscription(PoseStamped, "/goal_pose", self.on_goal_pose, 10)
        self.create_subscription(PoseStamped, "/move_base_simple/goal", self.on_goal_pose, 10)
        self.create_subscription(PoseStamped, "/start_pose", self.on_start_pose, 10)
        self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.on_initial_pose, 10)

        self.static_grid = [0 for _ in range(self.width * self.height)]
        self.raw_grid = [0 for _ in range(self.width * self.height)]
        self.grid = [0 for _ in range(self.width * self.height)]

        self.build_static_map()
        self.rebuild_grid_with_dynamic_obstacles()

        self.timer = self.create_timer(self.timer_period, self.publish_all)

        self.get_logger().info(
            f"A* demo started: {self.width}x{self.height}, resolution={self.resolution}"
        )

    def parse_dynamic_obstacles(self, text):
        obstacles = []

        for item in str(text).split(";"):
            item = item.strip()
            if not item:
                continue

            values = [float(v.strip()) for v in item.split(",")]
            if len(values) != 8:
                self.get_logger().warn(f"Invalid dynamic obstacle config: {item}")
                continue

            x, y, vx, vy, min_x, max_x, min_y, max_y = values
            obstacles.append({
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
            })

        return obstacles

    def index(self, x, y):
        return y * self.width + x

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def set_static_cell(self, x, y, value):
        if self.in_bounds(x, y):
            self.static_grid[self.index(x, y)] = value

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

    def clear_area(self, wx, wy, radius, target_grid):
        cx, cy = self.world_to_grid(wx, wy)
        r = int(radius / self.resolution)

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx = cx + dx
                ny = cy + dy
                if self.in_bounds(nx, ny):
                    target_grid[self.index(nx, ny)] = 0

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

    def build_static_map(self):
        self.static_grid = [0 for _ in range(self.width * self.height)]

        for x in range(self.width):
            self.set_static_cell(x, 0, 100)
            self.set_static_cell(x, self.height - 1, 100)

        for y in range(self.height):
            self.set_static_cell(0, y, 100)
            self.set_static_cell(self.width - 1, y, 100)

        route_world = [
            (0.8, 0.8),
            (2.2, 0.8),
            (4.5, 1.0),
            (4.5, 3.8),
            (6.7, 4.2),
            (7.5, 5.6),
            (9.3, 6.7),
            (11.0, 7.5),
            (12.8, 8.0),
        ]

        protected = set()
        route_cells = [self.world_to_grid(x, y) for x, y in route_world]

        for i in range(len(route_cells) - 1):
            for cx, cy in self.bresenham(route_cells[i], route_cells[i + 1]):
                for dy in range(-self.route_corridor_cells, self.route_corridor_cells + 1):
                    for dx in range(-self.route_corridor_cells, self.route_corridor_cells + 1):
                        nx = cx + dx
                        ny = cy + dy
                        if self.in_bounds(nx, ny):
                            protected.add((nx, ny))

        rng = random.Random(self.random_seed)

        for _ in range(self.random_obstacle_count):
            block_w = rng.randint(self.block_min_w, self.block_max_w)
            block_h = rng.randint(self.block_min_h, self.block_max_h)

            x0 = rng.randint(2, self.width - block_w - 3)
            y0 = rng.randint(2, self.height - block_h - 3)

            blocked = False
            for y in range(y0, y0 + block_h):
                for x in range(x0, x0 + block_w):
                    if (x, y) in protected:
                        blocked = True
                        break
                if blocked:
                    break

            if blocked:
                continue

            for y in range(y0, y0 + block_h):
                for x in range(x0, x0 + block_w):
                    self.set_static_cell(x, y, 100)

        for x, y in protected:
            if 0 < x < self.width - 1 and 0 < y < self.height - 1:
                self.static_grid[self.index(x, y)] = 0

        self.clear_area(self.start_world[0], self.start_world[1], 0.7, self.static_grid)
        self.clear_area(self.goal_world[0], self.goal_world[1], 0.7, self.static_grid)

    def update_dynamic_obstacles(self, dt):
        for obs in self.dynamic_obstacles:
            obs["x"] += obs["vx"] * dt
            obs["y"] += obs["vy"] * dt

            if obs["x"] < obs["min_x"] or obs["x"] > obs["max_x"]:
                obs["vx"] *= -1.0
                obs["x"] = max(obs["min_x"], min(obs["max_x"], obs["x"]))

            if obs["y"] < obs["min_y"] or obs["y"] > obs["max_y"]:
                obs["vy"] *= -1.0
                obs["y"] = max(obs["min_y"], min(obs["max_y"], obs["y"]))

    def rebuild_grid_with_dynamic_obstacles(self):
        self.raw_grid = self.static_grid.copy()

        for obs in self.dynamic_obstacles:
            cx, cy = self.world_to_grid(obs["x"], obs["y"])
            radius = int(self.dynamic_obstacle_radius / self.resolution)

            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    nx = cx + dx
                    ny = cy + dy
                    if not self.in_bounds(nx, ny):
                        continue
                    if math.hypot(dx, dy) <= radius:
                        self.raw_grid[self.index(nx, ny)] = 100

        self.clear_area(self.start_world[0], self.start_world[1], 0.35, self.raw_grid)
        self.clear_area(self.goal_world[0], self.goal_world[1], 0.35, self.raw_grid)
        self.inflate_obstacles()

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
        for r in range(1, 30):
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
        cells = [current]
        while current in came_from:
            current = came_from[current]
            cells.append(current)
        cells.reverse()
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

    def make_path_msg(self, cells):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        for gx, gy in cells:
            wx, wy = self.grid_to_world(gx, gy)

            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        return msg

    def publish_paths(self):
        start = self.world_to_grid(*self.start_world)
        goal = self.world_to_grid(*self.goal_world)

        raw_cells = self.astar(start, goal)
        simple_cells = self.simplify_path(raw_cells)

        self.raw_path_pub.publish(self.make_path_msg(raw_cells))
        self.path_pub.publish(self.make_path_msg(simple_cells))

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

    def publish_pose(self, publisher, xy):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = xy[0]
        msg.pose.position.y = xy[1]
        msg.pose.orientation.w = 1.0
        publisher.publish(msg)

    def publish_dynamic_obstacles(self):
        marker_array = MarkerArray()

        for i, obs in enumerate(self.dynamic_obstacles):
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = "map"
            marker.ns = "dynamic_obstacles"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = obs["x"]
            marker.pose.position.y = obs["y"]
            marker.pose.position.z = 0.15
            marker.pose.orientation.z = 0.3826834
            marker.pose.orientation.w = 0.9238795

            marker.scale.x = self.dynamic_obstacle_radius
            marker.scale.y = self.dynamic_obstacle_radius
            marker.scale.z = 0.25

            marker.color.r = 1.0
            marker.color.g = 0.65
            marker.color.b = 0.05
            marker.color.a = 0.95

            marker_array.markers.append(marker)

        self.dynamic_pub.publish(marker_array)

    def publish_all(self):
        self.update_dynamic_obstacles(self.timer_period)
        self.rebuild_grid_with_dynamic_obstacles()

        self.publish_map()
        self.publish_dynamic_obstacles()
        self.publish_pose(self.start_pub, self.start_world)
        self.publish_pose(self.goal_pub, self.goal_world)
        self.publish_paths()

    def on_goal_pose(self, msg):
        self.goal_world = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f"New goal: {self.goal_world[0]:.2f}, {self.goal_world[1]:.2f}")

    def on_start_pose(self, msg):
        self.start_world = (msg.pose.position.x, msg.pose.position.y)

    def on_initial_pose(self, msg):
        self.start_world = (msg.pose.pose.position.x, msg.pose.pose.position.y)


def main():
    rclpy.init()
    node = AStarPlannerDemo()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()