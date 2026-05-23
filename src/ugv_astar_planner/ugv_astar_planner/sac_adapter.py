import json
import math
import time
from pathlib import Path as FilePath

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path as NavPath
from std_msgs.msg import Float32, String
from visualization_msgs.msg import MarkerArray


DEFAULT_STATE_FEATURES = [
    "goal_distance_m",
    "min_dynamic_obstacle_distance_m",
    "safety_scale",
    "raw_linear_mps",
    "safe_linear_mps",
    "global_path_length_m",
    "local_trajectory_length_m",
    "progress_mps",
    "risk",
]

DEFAULT_ACTION_FEATURES = [
    "speed_scale",
    "lookahead_scale",
    "local_horizon_scale",
    "obstacle_caution",
]

DEFAULT_ACTION_LOW = [0.62, 0.95, 1.00, 1.00]
DEFAULT_ACTION_HIGH = [1.00, 1.28, 1.45, 1.70]


class SacAdapter(Node):
    def __init__(self):
        super().__init__("sac_adapter")

        self.declare_parameter("update_period", 0.2)
        self.declare_parameter("mode", "b4_v2_proactive_rule_policy")
        self.declare_parameter("model_path", "/root/ugv_nav_ws/results/c2_offline_sac/sac_actor.pt")
        self.declare_parameter("torch_device", "cpu")
        self.declare_parameter("fallback_to_rule", True)

        self.declare_parameter("safety_trigger_enabled", False)
        self.declare_parameter("safety_trigger_obstacle_distance", 0.80)
        self.declare_parameter("safety_trigger_speed_threshold", 0.88)
        self.declare_parameter("safety_trigger_caution_threshold", 1.25)
        self.declare_parameter("safety_trigger_speed_scale", 0.85)
        self.declare_parameter("safety_trigger_lookahead_scale", 1.10)
        self.declare_parameter("safety_trigger_local_horizon_scale", 1.20)
        self.declare_parameter("safety_trigger_obstacle_caution", 1.35)

        self.declare_parameter("caution_distance", 1.40)
        self.declare_parameter("warning_distance", 0.75)
        self.declare_parameter("emergency_distance", 0.40)
        self.declare_parameter("speed_floor", 0.62)
        self.declare_parameter("speed_ceiling", 1.0)
        self.declare_parameter("lookahead_ceiling", 1.28)
        self.declare_parameter("local_horizon_ceiling", 1.45)
        self.declare_parameter("obstacle_caution_ceiling", 1.70)
        self.declare_parameter("rise_alpha", 0.55)
        self.declare_parameter("fall_alpha", 0.22)

        self.declare_parameter("risk_gate_enabled", False)
        self.declare_parameter("risk_gate_clear_distance", 1.20)
        self.declare_parameter("risk_gate_warning_distance", 0.60)
        self.declare_parameter("risk_gate_clear_speed_floor", 0.95)
        self.declare_parameter("risk_gate_warning_speed_floor", 0.80)
        self.declare_parameter("risk_gate_clear_caution_ceiling", 1.10)
        self.declare_parameter("risk_gate_warning_caution_ceiling", 1.35)
        self.declare_parameter("risk_gate_clear_lookahead_ceiling", 1.08)
        self.declare_parameter("risk_gate_warning_lookahead_ceiling", 1.18)
        self.declare_parameter("risk_gate_clear_horizon_ceiling", 1.10)
        self.declare_parameter("risk_gate_warning_horizon_ceiling", 1.25)

        self.declare_parameter("mode_selector_enabled", False)
        self.declare_parameter("mode_selector_clear_distance", 1.20)
        self.declare_parameter("mode_selector_warning_distance", 0.60)
        self.declare_parameter("mode_selector_cautious_speed_trigger", 0.88)
        self.declare_parameter("mode_selector_emergency_speed_trigger", 0.78)
        self.declare_parameter("mode_selector_cautious_caution_trigger", 1.25)
        self.declare_parameter("mode_selector_emergency_caution_trigger", 1.45)

        self.declare_parameter("mode_normal_speed", 1.00)
        self.declare_parameter("mode_normal_lookahead", 1.00)
        self.declare_parameter("mode_normal_horizon", 1.00)
        self.declare_parameter("mode_normal_caution", 1.00)

        self.declare_parameter("mode_cautious_speed", 0.90)
        self.declare_parameter("mode_cautious_lookahead", 1.08)
        self.declare_parameter("mode_cautious_horizon", 1.15)
        self.declare_parameter("mode_cautious_caution", 1.25)

        self.declare_parameter("mode_emergency_speed", 0.72)
        self.declare_parameter("mode_emergency_lookahead", 1.18)
        self.declare_parameter("mode_emergency_horizon", 1.35)
        self.declare_parameter("mode_emergency_caution", 1.55)

        self.update_period = float(self.get_parameter("update_period").value)
        self.mode = str(self.get_parameter("mode").value)
        self.model_path = str(self.get_parameter("model_path").value)
        self.torch_device_name = str(self.get_parameter("torch_device").value)
        self.fallback_to_rule = bool(self.get_parameter("fallback_to_rule").value)

        self.safety_trigger_enabled = bool(self.get_parameter("safety_trigger_enabled").value)
        self.safety_trigger_obstacle_distance = float(self.get_parameter("safety_trigger_obstacle_distance").value)
        self.safety_trigger_speed_threshold = float(self.get_parameter("safety_trigger_speed_threshold").value)
        self.safety_trigger_caution_threshold = float(self.get_parameter("safety_trigger_caution_threshold").value)
        self.safety_trigger_speed_scale = float(self.get_parameter("safety_trigger_speed_scale").value)
        self.safety_trigger_lookahead_scale = float(self.get_parameter("safety_trigger_lookahead_scale").value)
        self.safety_trigger_local_horizon_scale = float(self.get_parameter("safety_trigger_local_horizon_scale").value)
        self.safety_trigger_obstacle_caution = float(self.get_parameter("safety_trigger_obstacle_caution").value)

        self.caution_distance = float(self.get_parameter("caution_distance").value)
        self.warning_distance = float(self.get_parameter("warning_distance").value)
        self.emergency_distance = float(self.get_parameter("emergency_distance").value)

        self.speed_floor = float(self.get_parameter("speed_floor").value)
        self.speed_ceiling = float(self.get_parameter("speed_ceiling").value)
        self.lookahead_ceiling = float(self.get_parameter("lookahead_ceiling").value)
        self.local_horizon_ceiling = float(self.get_parameter("local_horizon_ceiling").value)
        self.obstacle_caution_ceiling = float(self.get_parameter("obstacle_caution_ceiling").value)

        self.rise_alpha = float(self.get_parameter("rise_alpha").value)
        self.fall_alpha = float(self.get_parameter("fall_alpha").value)

        self.risk_gate_enabled = bool(self.get_parameter("risk_gate_enabled").value)
        self.risk_gate_clear_distance = float(self.get_parameter("risk_gate_clear_distance").value)
        self.risk_gate_warning_distance = float(self.get_parameter("risk_gate_warning_distance").value)
        self.risk_gate_clear_speed_floor = float(self.get_parameter("risk_gate_clear_speed_floor").value)
        self.risk_gate_warning_speed_floor = float(self.get_parameter("risk_gate_warning_speed_floor").value)
        self.risk_gate_clear_caution_ceiling = float(self.get_parameter("risk_gate_clear_caution_ceiling").value)
        self.risk_gate_warning_caution_ceiling = float(self.get_parameter("risk_gate_warning_caution_ceiling").value)
        self.risk_gate_clear_lookahead_ceiling = float(self.get_parameter("risk_gate_clear_lookahead_ceiling").value)
        self.risk_gate_warning_lookahead_ceiling = float(self.get_parameter("risk_gate_warning_lookahead_ceiling").value)
        self.risk_gate_clear_horizon_ceiling = float(self.get_parameter("risk_gate_clear_horizon_ceiling").value)
        self.risk_gate_warning_horizon_ceiling = float(self.get_parameter("risk_gate_warning_horizon_ceiling").value)

        self.mode_selector_enabled = bool(self.get_parameter("mode_selector_enabled").value)
        self.mode_selector_clear_distance = float(self.get_parameter("mode_selector_clear_distance").value)
        self.mode_selector_warning_distance = float(self.get_parameter("mode_selector_warning_distance").value)
        self.mode_selector_cautious_speed_trigger = float(self.get_parameter("mode_selector_cautious_speed_trigger").value)
        self.mode_selector_emergency_speed_trigger = float(self.get_parameter("mode_selector_emergency_speed_trigger").value)
        self.mode_selector_cautious_caution_trigger = float(self.get_parameter("mode_selector_cautious_caution_trigger").value)
        self.mode_selector_emergency_caution_trigger = float(self.get_parameter("mode_selector_emergency_caution_trigger").value)

        self.mode_normal_speed = float(self.get_parameter("mode_normal_speed").value)
        self.mode_normal_lookahead = float(self.get_parameter("mode_normal_lookahead").value)
        self.mode_normal_horizon = float(self.get_parameter("mode_normal_horizon").value)
        self.mode_normal_caution = float(self.get_parameter("mode_normal_caution").value)

        self.mode_cautious_speed = float(self.get_parameter("mode_cautious_speed").value)
        self.mode_cautious_lookahead = float(self.get_parameter("mode_cautious_lookahead").value)
        self.mode_cautious_horizon = float(self.get_parameter("mode_cautious_horizon").value)
        self.mode_cautious_caution = float(self.get_parameter("mode_cautious_caution").value)

        self.mode_emergency_speed = float(self.get_parameter("mode_emergency_speed").value)
        self.mode_emergency_lookahead = float(self.get_parameter("mode_emergency_lookahead").value)
        self.mode_emergency_horizon = float(self.get_parameter("mode_emergency_horizon").value)
        self.mode_emergency_caution = float(self.get_parameter("mode_emergency_caution").value)

        self.robot_pose = None
        self.goal_pose = None
        self.dynamic_obstacles = []
        self.global_path = []
        self.local_path = []

        self.goal_distance_m = -1.0
        self.min_obstacle_distance_m = -1.0
        self.safety_scale = 1.0
        self.raw_linear_mps = 0.0
        self.safe_linear_mps = 0.0

        self.speed_scale = 1.0
        self.lookahead_scale = 1.0
        self.local_horizon_scale = 1.0
        self.obstacle_caution = 1.0

        self.prev_goal_distance = None
        self.prev_stamp = time.time()
        self.progress_mps = 0.0

        self.torch = None
        self.actor = None
        self.policy_ready = False
        self.policy_error = ""
        self.policy_state_features = DEFAULT_STATE_FEATURES
        self.policy_action_features = DEFAULT_ACTION_FEATURES
        self.policy_action_low = DEFAULT_ACTION_LOW
        self.policy_action_high = DEFAULT_ACTION_HIGH
        self.policy_state_mean = [0.0] * len(DEFAULT_STATE_FEATURES)
        self.policy_state_std = [1.0] * len(DEFAULT_STATE_FEATURES)

        self.use_torch_policy = "torch" in self.mode.lower() or "actor" in self.mode.lower()
        if self.use_torch_policy:
            self.load_torch_policy()

        self.create_subscription(PoseStamped, "/robot_pose", self.on_robot_pose, 10)
        self.create_subscription(PoseStamped, "/astar_goal", self.on_goal_pose, 10)
        self.create_subscription(MarkerArray, "/dynamic_obstacles", self.on_dynamic_obstacles, 10)
        self.create_subscription(NavPath, "/astar_path", self.on_global_path, 10)
        self.create_subscription(NavPath, "/local_trajectory", self.on_local_path, 10)
        self.create_subscription(Twist, "/cmd_vel_raw", self.on_cmd_raw, 10)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd_safe, 10)

        self.create_subscription(Float32, "/metrics/goal_distance", self.on_metric_goal_distance, 10)
        self.create_subscription(Float32, "/metrics/min_dynamic_obstacle_distance", self.on_metric_min_obstacle, 10)
        self.create_subscription(Float32, "/metrics/safety_scale", self.on_metric_safety_scale, 10)

        self.adaptation_pub = self.create_publisher(String, "/sac/adaptation", 10)
        self.state_pub = self.create_publisher(String, "/sac/state", 10)
        self.reward_pub = self.create_publisher(Float32, "/sac/reward", 10)
        self.policy_status_pub = self.create_publisher(String, "/sac/policy_status", 10)

        self.timer = self.create_timer(self.update_period, self.on_timer)

        if self.use_torch_policy and self.policy_ready:
            self.get_logger().info(f"SAC adapter started in torch_policy mode: {self.model_path}")
        elif self.use_torch_policy:
            self.get_logger().warn(f"Torch policy requested but unavailable: {self.policy_error}")
        else:
            self.get_logger().info("SAC adapter started in B4-v2 rule policy mode.")

    def build_torch_actor_class(self, torch, nn):
        def mlp(in_dim, out_dim, hidden_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )

        class TorchActor(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_dim):
                super().__init__()
                self.backbone = mlp(state_dim, hidden_dim, hidden_dim)
                self.mean = nn.Linear(hidden_dim, action_dim)
                self.log_std = nn.Linear(hidden_dim, action_dim)

            def forward(self, state):
                h = self.backbone[:-1](state)
                h = self.backbone[-1](h)
                mean = self.mean(h)
                log_std = self.log_std(h).clamp(-5.0, 2.0)
                return mean, log_std

            def deterministic(self, state):
                mean, _ = self(state)
                return torch.tanh(mean)

        return TorchActor

    def load_torch_policy(self):
        try:
            import torch
            from torch import nn
        except Exception as exc:
            self.policy_error = f"failed to import torch: {exc}"
            return

        path = FilePath(self.model_path).expanduser()
        if not path.exists():
            self.policy_error = f"model file not found: {path}"
            return

        try:
            device = torch.device(self.torch_device_name)
            try:
                checkpoint = torch.load(str(path), map_location=device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(str(path), map_location=device)

            metadata = checkpoint.get("metadata", {})
            self.policy_state_features = metadata.get("state_features", DEFAULT_STATE_FEATURES)
            self.policy_action_features = metadata.get("action_features", DEFAULT_ACTION_FEATURES)
            self.policy_action_low = metadata.get("action_low", DEFAULT_ACTION_LOW)
            self.policy_action_high = metadata.get("action_high", DEFAULT_ACTION_HIGH)
            self.policy_state_mean = metadata.get(
                "state_mean", [0.0] * len(self.policy_state_features)
            )
            self.policy_state_std = metadata.get(
                "state_std", [1.0] * len(self.policy_state_features)
            )

            hidden_dim = int(metadata.get("hidden_dim", 128))
            state_dim = len(self.policy_state_features)
            action_dim = len(self.policy_action_features)

            TorchActor = self.build_torch_actor_class(torch, nn)
            actor = TorchActor(state_dim, action_dim, hidden_dim).to(device)
            actor.load_state_dict(checkpoint["actor_state_dict"])
            actor.eval()

            self.torch = torch
            self.torch_device = device
            self.actor = actor
            self.policy_ready = True
            self.policy_error = ""
        except Exception as exc:
            self.policy_ready = False
            self.policy_error = f"failed to load torch policy: {exc}"

    def on_robot_pose(self, msg):
        self.robot_pose = msg

    def on_goal_pose(self, msg):
        self.goal_pose = msg

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

    def on_global_path(self, msg):
        self.global_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def on_local_path(self, msg):
        self.local_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    def on_cmd_raw(self, msg):
        self.raw_linear_mps = msg.linear.x

    def on_cmd_safe(self, msg):
        self.safe_linear_mps = msg.linear.x

    def on_metric_goal_distance(self, msg):
        self.goal_distance_m = float(msg.data)

    def on_metric_min_obstacle(self, msg):
        self.min_obstacle_distance_m = float(msg.data)

    def on_metric_safety_scale(self, msg):
        self.safety_scale = max(0.0, min(1.0, float(msg.data)))

    def path_length(self, points):
        if len(points) < 2:
            return 0.0
        return sum(
            math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
            for i in range(len(points) - 1)
        )

    def fallback_goal_distance(self):
        if self.robot_pose is None or self.goal_pose is None:
            return -1.0
        rx = self.robot_pose.pose.position.x
        ry = self.robot_pose.pose.position.y
        gx = self.goal_pose.pose.position.x
        gy = self.goal_pose.pose.position.y
        return math.hypot(gx - rx, gy - ry)

    def fallback_min_obstacle_distance(self):
        if self.robot_pose is None or not self.dynamic_obstacles:
            return -1.0
        rx = self.robot_pose.pose.position.x
        ry = self.robot_pose.pose.position.y
        nearest = None
        for ox, oy, radius in self.dynamic_obstacles:
            d = math.hypot(ox - rx, oy - ry) - radius
            if nearest is None or d < nearest:
                nearest = d
        return nearest if nearest is not None else -1.0

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def blend(self, old, target):
        alpha = self.rise_alpha if abs(target - 1.0) > abs(old - 1.0) else self.fall_alpha
        return old + alpha * (target - old)

    def compute_rule_action(self, obstacle_distance):
        if obstacle_distance < 0.0:
            return 1.0, 1.0, 1.0, 1.0, 0.0

        if obstacle_distance >= self.caution_distance:
            risk = 0.0
            speed = 1.0
            lookahead = 1.0
            horizon = 1.0
            caution = 1.0
        elif obstacle_distance >= self.warning_distance:
            r = (self.caution_distance - obstacle_distance) / max(
                1e-6, self.caution_distance - self.warning_distance
            )
            risk = 0.35 * r
            speed = 1.0 - 0.10 * r
            lookahead = 1.0 + 0.20 * r
            horizon = 1.0 + 0.25 * r
            caution = 1.0 + 0.35 * r
        elif obstacle_distance >= self.emergency_distance:
            r = (self.warning_distance - obstacle_distance) / max(
                1e-6, self.warning_distance - self.emergency_distance
            )
            risk = 0.35 + 0.45 * r
            speed = 0.90 - 0.18 * r
            lookahead = 1.20 + 0.08 * r
            horizon = 1.25 + 0.18 * r
            caution = 1.35 + 0.30 * r
        else:
            risk = 1.0
            speed = self.speed_floor
            lookahead = 1.12
            horizon = self.local_horizon_ceiling
            caution = self.obstacle_caution_ceiling

        speed, lookahead, horizon, caution = self.apply_safety_bounds(
            speed, lookahead, horizon, caution
        )
        return speed, lookahead, horizon, caution, risk

    def apply_safety_bounds(self, speed, lookahead, horizon, caution):
        if self.safety_scale < 0.65:
            speed = min(speed, 0.82)
            horizon = max(horizon, 1.30)
            caution = max(caution, 1.45)

        speed = self.clamp(speed, self.speed_floor, self.speed_ceiling)
        lookahead = self.clamp(lookahead, 0.95, self.lookahead_ceiling)
        horizon = self.clamp(horizon, 1.0, self.local_horizon_ceiling)
        caution = self.clamp(caution, 1.0, self.obstacle_caution_ceiling)

        return speed, lookahead, horizon, caution

    def apply_mode_selector(self, speed, lookahead, horizon, caution, obstacle_distance):
        if not self.mode_selector_enabled:
            return speed, lookahead, horizon, caution, "disabled"

        if obstacle_distance < 0.0 or obstacle_distance >= self.mode_selector_clear_distance:
            mode = "normal"
        elif obstacle_distance >= self.mode_selector_warning_distance:
            if (
                speed <= self.mode_selector_cautious_speed_trigger
                or caution >= self.mode_selector_cautious_caution_trigger
            ):
                mode = "cautious"
            else:
                mode = "normal"
        else:
            if (
                speed <= self.mode_selector_emergency_speed_trigger
                or caution >= self.mode_selector_emergency_caution_trigger
            ):
                mode = "emergency"
            else:
                mode = "cautious"

        if mode == "normal":
            speed = self.mode_normal_speed
            lookahead = self.mode_normal_lookahead
            horizon = self.mode_normal_horizon
            caution = self.mode_normal_caution
        elif mode == "cautious":
            speed = self.mode_cautious_speed
            lookahead = self.mode_cautious_lookahead
            horizon = self.mode_cautious_horizon
            caution = self.mode_cautious_caution
        else:
            speed = self.mode_emergency_speed
            lookahead = self.mode_emergency_lookahead
            horizon = self.mode_emergency_horizon
            caution = self.mode_emergency_caution

        speed, lookahead, horizon, caution = self.apply_safety_bounds(
            speed, lookahead, horizon, caution
        )
        return speed, lookahead, horizon, caution, mode

    def apply_risk_gate(self, speed, lookahead, horizon, caution, obstacle_distance):
        if not self.risk_gate_enabled:
            return speed, lookahead, horizon, caution, "disabled"

        if obstacle_distance < 0.0:
            return speed, lookahead, horizon, caution, "unknown_obstacle"

        if self.safety_scale < 0.65:
            return speed, lookahead, horizon, caution, "safety_filter_override"

        if obstacle_distance >= self.risk_gate_clear_distance:
            speed = max(speed, self.risk_gate_clear_speed_floor)
            lookahead = min(lookahead, self.risk_gate_clear_lookahead_ceiling)
            horizon = min(horizon, self.risk_gate_clear_horizon_ceiling)
            caution = min(caution, self.risk_gate_clear_caution_ceiling)
            mode = "clear_path"
        elif obstacle_distance >= self.risk_gate_warning_distance:
            speed = max(speed, self.risk_gate_warning_speed_floor)
            lookahead = min(lookahead, self.risk_gate_warning_lookahead_ceiling)
            horizon = min(horizon, self.risk_gate_warning_horizon_ceiling)
            caution = min(caution, self.risk_gate_warning_caution_ceiling)
            mode = "warning_zone"
        else:
            mode = "high_risk_zone"

        speed, lookahead, horizon, caution = self.apply_safety_bounds(
            speed, lookahead, horizon, caution
        )
        return speed, lookahead, horizon, caution, mode

    def compute_torch_action(self, state):
        if not self.policy_ready or self.actor is None:
            raise RuntimeError(self.policy_error or "torch policy is not ready")

        values = []
        for name, mean, std in zip(
            self.policy_state_features,
            self.policy_state_mean,
            self.policy_state_std,
        ):
            std = std if abs(std) > 1e-6 else 1.0
            values.append((float(state.get(name, 0.0)) - float(mean)) / float(std))

        torch = self.torch
        with torch.no_grad():
            tensor = torch.tensor([values], dtype=torch.float32, device=self.torch_device)
            action_norm = self.actor.deterministic(tensor).cpu().numpy()[0].tolist()

        action = []
        for value, low, high in zip(action_norm, self.policy_action_low, self.policy_action_high):
            real_value = float(low) + 0.5 * (float(value) + 1.0) * (float(high) - float(low))
            action.append(real_value)

        action_map = dict(zip(self.policy_action_features, action))

        speed = action_map.get("speed_scale", 1.0)
        lookahead = action_map.get("lookahead_scale", 1.0)
        horizon = action_map.get("local_horizon_scale", 1.0)
        caution = action_map.get("obstacle_caution", 1.0)

        return self.apply_safety_bounds(speed, lookahead, horizon, caution)

    def compute_safety_trigger_action(self, obstacle_distance, torch_action):
        torch_speed, torch_lookahead, torch_horizon, torch_caution = torch_action

        real_risk = (
            obstacle_distance >= 0.0
            and obstacle_distance < self.safety_trigger_obstacle_distance
        )
        rl_intent = (
            torch_speed < self.safety_trigger_speed_threshold
            or torch_caution > self.safety_trigger_caution_threshold
        )
        trigger_active = bool(real_risk and rl_intent)

        if trigger_active:
            target = (
                self.safety_trigger_speed_scale,
                self.safety_trigger_lookahead_scale,
                self.safety_trigger_local_horizon_scale,
                self.safety_trigger_obstacle_caution,
            )
            reason = "safe_mode"
        else:
            target = (1.0, 1.0, 1.0, 1.0)
            reason = "baseline_clear" if not real_risk else "baseline_no_rl_intent"

        trigger_info = {
            "enabled": True,
            "active": trigger_active,
            "real_risk": bool(real_risk),
            "rl_intent": bool(rl_intent),
            "reason": reason,
            "obstacle_distance_m": round(obstacle_distance, 3),
            "obstacle_threshold_m": round(self.safety_trigger_obstacle_distance, 3),
            "speed_threshold": round(self.safety_trigger_speed_threshold, 3),
            "caution_threshold": round(self.safety_trigger_caution_threshold, 3),
            "torch_action": {
                "speed_scale": round(torch_speed, 3),
                "lookahead_scale": round(torch_lookahead, 3),
                "local_horizon_scale": round(torch_horizon, 3),
                "obstacle_caution": round(torch_caution, 3),
            },
        }

        return target, trigger_info

    def compute_progress(self, goal_distance):
        now = time.time()
        dt = max(1e-6, now - self.prev_stamp)

        if goal_distance >= 0.0 and self.prev_goal_distance is not None:
            self.progress_mps = (self.prev_goal_distance - goal_distance) / dt

        if goal_distance >= 0.0:
            self.prev_goal_distance = goal_distance

        self.prev_stamp = now

    def publish_string(self, publisher, data):
        msg = String()
        msg.data = data
        publisher.publish(msg)

    def publish_reward(self, reward):
        msg = Float32()
        msg.data = float(reward)
        self.reward_pub.publish(msg)

    def publish_policy_status(self, policy_source, trigger_info=None):
        status = {
            "mode": self.mode,
            "requested_torch_policy": self.use_torch_policy,
            "policy_ready": self.policy_ready,
            "policy_source": policy_source,
            "model_path": self.model_path,
            "error": self.policy_error,
            "safety_trigger_enabled": self.safety_trigger_enabled,
        }
        if trigger_info is not None:
            status["safety_trigger"] = trigger_info
        self.publish_string(self.policy_status_pub, json.dumps(status, separators=(",", ":")))

    def on_timer(self):
        goal_distance = self.goal_distance_m
        if goal_distance < 0.0:
            goal_distance = self.fallback_goal_distance()

        obstacle_distance = self.min_obstacle_distance_m
        if obstacle_distance < 0.0:
            obstacle_distance = self.fallback_min_obstacle_distance()

        self.compute_progress(goal_distance)

        global_len = self.path_length(self.global_path)
        local_len = self.path_length(self.local_path)

        rule_speed, rule_lookahead, rule_horizon, rule_caution, risk = self.compute_rule_action(
            obstacle_distance
        )

        state = {
            "schema_version": 1,
            "goal_distance_m": round(goal_distance, 3),
            "min_dynamic_obstacle_distance_m": round(obstacle_distance, 3),
            "safety_scale": round(self.safety_scale, 3),
            "raw_linear_mps": round(self.raw_linear_mps, 3),
            "safe_linear_mps": round(self.safe_linear_mps, 3),
            "progress_mps": round(self.progress_mps, 3),
            "global_path_length_m": round(global_len, 3),
            "local_trajectory_length_m": round(local_len, 3),
            "risk": round(risk, 3),
        }

        policy_source = "rule_policy"

        if self.use_torch_policy and self.policy_ready:
            try:
                target_speed, target_lookahead, target_horizon, target_caution = self.compute_torch_action(state)
                policy_source = "torch_policy"
            except Exception as exc:
                self.policy_error = str(exc)
                if not self.fallback_to_rule:
                    target_speed, target_lookahead, target_horizon, target_caution = 0.0, 1.0, 1.0, 1.0
                    policy_source = "torch_policy_error_stop"
                else:
                    target_speed, target_lookahead, target_horizon, target_caution = (
                        rule_speed,
                        rule_lookahead,
                        rule_horizon,
                        rule_caution,
                    )
                    policy_source = "torch_policy_error_fallback_rule"
        else:
            target_speed, target_lookahead, target_horizon, target_caution = (
                rule_speed,
                rule_lookahead,
                rule_horizon,
                rule_caution,
            )

        raw_target_action = {
            "speed_scale": round(target_speed, 3),
            "lookahead_scale": round(target_lookahead, 3),
            "local_horizon_scale": round(target_horizon, 3),
            "obstacle_caution": round(target_caution, 3),
        }

        target_speed, target_lookahead, target_horizon, target_caution, mode_selector_mode = (
            self.apply_mode_selector(
                target_speed,
                target_lookahead,
                target_horizon,
                target_caution,
                obstacle_distance,
            )
        )

        mode_selector_action = {
            "speed_scale": round(target_speed, 3),
            "lookahead_scale": round(target_lookahead, 3),
            "local_horizon_scale": round(target_horizon, 3),
            "obstacle_caution": round(target_caution, 3),
        }

        target_speed, target_lookahead, target_horizon, target_caution, risk_gate_mode = (
            self.apply_risk_gate(
                target_speed,
                target_lookahead,
                target_horizon,
                target_caution,
                obstacle_distance,
            )
        )

        gated_target_action = {
            "speed_scale": round(target_speed, 3),
            "lookahead_scale": round(target_lookahead, 3),
            "local_horizon_scale": round(target_horizon, 3),
            "obstacle_caution": round(target_caution, 3),
        }

        trigger_info = {
            "enabled": self.safety_trigger_enabled,
            "active": False,
            "real_risk": False,
            "rl_intent": False,
            "reason": "disabled",
            "torch_action": {},
        }

        if self.safety_trigger_enabled:
            if self.use_torch_policy and self.policy_ready:
                try:
                    torch_action = self.compute_torch_action(state)
                    (
                        target_speed,
                        target_lookahead,
                        target_horizon,
                        target_caution,
                    ), trigger_info = self.compute_safety_trigger_action(
                        obstacle_distance, torch_action
                    )
                    policy_source = (
                        "safety_trigger_safe_mode"
                        if trigger_info["active"]
                        else "safety_trigger_baseline"
                    )
                except Exception as exc:
                    self.policy_error = str(exc)
                    target_speed, target_lookahead, target_horizon, target_caution = (
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    )
                    policy_source = "safety_trigger_error_baseline"
                    trigger_info = {
                        "enabled": True,
                        "active": False,
                        "real_risk": False,
                        "rl_intent": False,
                        "reason": "torch_error_baseline",
                        "error": self.policy_error,
                        "torch_action": {},
                    }
            else:
                target_speed, target_lookahead, target_horizon, target_caution = (
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                )
                policy_source = "safety_trigger_no_torch_baseline"
                trigger_info = {
                    "enabled": True,
                    "active": False,
                    "real_risk": False,
                    "rl_intent": False,
                    "reason": "torch_unavailable_baseline",
                    "error": self.policy_error,
                    "torch_action": {},
                }

        self.speed_scale = self.blend(self.speed_scale, target_speed)
        self.lookahead_scale = self.blend(self.lookahead_scale, target_lookahead)
        self.local_horizon_scale = self.blend(self.local_horizon_scale, target_horizon)
        self.obstacle_caution = self.blend(self.obstacle_caution, target_caution)

        reward = 0.2 * self.safe_linear_mps
        reward += 0.3 * max(0.0, self.progress_mps)
        reward -= 0.35 * risk
        reward -= 0.20 * max(0.0, 1.0 - self.safety_scale)
        if goal_distance >= 0.0:
            reward -= 0.01 * goal_distance
            if goal_distance < 0.35:
                reward += 1.0

        action = {
            "speed_scale": round(self.speed_scale, 3),
            "lookahead_scale": round(self.lookahead_scale, 3),
            "local_horizon_scale": round(self.local_horizon_scale, 3),
            "obstacle_caution": round(self.obstacle_caution, 3),
        }

        policy_type = (
            "rl_safety_trigger"
            if self.safety_trigger_enabled
            else ("torch_actor" if policy_source == "torch_policy" else "rl_parameter_interface")
        )

        adaptation = {
            "schema_version": 1,
            "mode": self.mode,
            "policy_type": policy_type,
            "policy_source": policy_source,
            "stamp": time.time(),
            "state": state,
            "action": action,
            "safety_trigger": trigger_info,
            "raw_target_action": raw_target_action,
            "mode_selector_enabled": self.mode_selector_enabled,
            "mode_selector_mode": mode_selector_mode,
            "mode_selector_action": mode_selector_action,
            "gated_target_action": gated_target_action,
            "risk_gate_enabled": self.risk_gate_enabled,
            "risk_gate_mode": risk_gate_mode,
            "risk": round(risk, 3),
            "reward": round(reward, 4),
            "speed_scale": action["speed_scale"],
            "lookahead_scale": action["lookahead_scale"],
            "local_horizon_scale": action["local_horizon_scale"],
            "obstacle_caution": action["obstacle_caution"],
        }

        self.publish_string(self.state_pub, json.dumps(state, separators=(",", ":")))
        self.publish_string(self.adaptation_pub, json.dumps(adaptation, separators=(",", ":")))
        self.publish_reward(reward)
        self.publish_policy_status(policy_source, trigger_info)


def main():
    rclpy.init()
    node = SacAdapter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
