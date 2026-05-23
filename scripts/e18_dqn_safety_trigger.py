#!/usr/bin/env python3
import csv
import json
import os
import random
import time
from collections import deque
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String


STATE_KEYS = [
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


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class DqnSafetyTrigger(Node):
    def __init__(self):
        super().__init__("e18_dqn_safety_trigger")

        self.declare_parameter("model_path", "/root/ugv_nav_ws/results/stage_e18_dqn_safety_trigger/dqn_trigger.pt")
        self.declare_parameter("log_path", "/root/ugv_nav_ws/results/stage_e18_dqn_safety_trigger/dqn_training_log.csv")
        self.declare_parameter("train_enabled", True)
        self.declare_parameter("eval_epsilon", 0.0)

        self.declare_parameter("update_period", 0.2)
        self.declare_parameter("gamma", 0.98)
        self.declare_parameter("lr", 0.0005)
        self.declare_parameter("batch_size", 64)
        self.declare_parameter("buffer_size", 50000)
        self.declare_parameter("warmup_steps", 300)
        self.declare_parameter("target_update_steps", 200)

        self.declare_parameter("epsilon_start", 1.0)
        self.declare_parameter("epsilon_min", 0.05)
        self.declare_parameter("epsilon_decay", 0.995)

        self.declare_parameter("safe_speed_scale", 0.85)
        self.declare_parameter("safe_lookahead_scale", 1.10)
        self.declare_parameter("safe_horizon_scale", 1.20)
        self.declare_parameter("safe_obstacle_caution", 1.35)

        self.model_path = Path(str(self.get_parameter("model_path").value))
        self.log_path = Path(str(self.get_parameter("log_path").value))
        self.train_enabled = bool(self.get_parameter("train_enabled").value)
        self.eval_epsilon = float(self.get_parameter("eval_epsilon").value)

        self.update_period = float(self.get_parameter("update_period").value)
        self.gamma = float(self.get_parameter("gamma").value)
        self.lr = float(self.get_parameter("lr").value)
        self.batch_size = int(self.get_parameter("batch_size").value)
        self.buffer_size = int(self.get_parameter("buffer_size").value)
        self.warmup_steps = int(self.get_parameter("warmup_steps").value)
        self.target_update_steps = int(self.get_parameter("target_update_steps").value)

        self.epsilon = float(self.get_parameter("epsilon_start").value)
        self.epsilon_min = float(self.get_parameter("epsilon_min").value)
        self.epsilon_decay = float(self.get_parameter("epsilon_decay").value)

        self.safe_action = {
            "speed_scale": float(self.get_parameter("safe_speed_scale").value),
            "lookahead_scale": float(self.get_parameter("safe_lookahead_scale").value),
            "local_horizon_scale": float(self.get_parameter("safe_horizon_scale").value),
            "obstacle_caution": float(self.get_parameter("safe_obstacle_caution").value),
        }
        self.base_action = {
            "speed_scale": 1.0,
            "lookahead_scale": 1.0,
            "local_horizon_scale": 1.0,
            "obstacle_caution": 1.0,
        }

        import torch
        import torch.nn as nn
        import torch.optim as optim

        self.torch = torch
        self.nn = nn

        self.q = nn.Sequential(
            nn.Linear(len(STATE_KEYS), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        self.target_q = nn.Sequential(
            nn.Linear(len(STATE_KEYS), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        self.target_q.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay = deque(maxlen=self.buffer_size)
        self.train_steps = 0
        self.total_steps = 0
        self.episode_id = 0
        self.prev_state_vec = None
        self.prev_action = None
        self.prev_goal_distance = None
        self.prev_time = None
        self.latest_summary = None
        self.done_latched = False

        self.load_checkpoint()

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, "a", newline="")
        self.log_writer = csv.DictWriter(self.log_file, fieldnames=[
            "time_s", "episode_id", "step", "action", "reward", "done",
            "goal_distance_m", "min_obstacle_m", "progress_mps",
            "epsilon", "loss", "q_baseline", "q_safe",
        ])
        if self.log_path.stat().st_size == 0:
            self.log_writer.writeheader()
            self.log_file.flush()

        self.create_subscription(String, "/metrics/summary", self.on_summary, 10)

        self.adaptation_pub = self.create_publisher(String, "/sac/adaptation", 10)
        self.state_pub = self.create_publisher(String, "/sac/state", 10)
        self.reward_pub = self.create_publisher(Float32, "/sac/reward", 10)
        self.status_pub = self.create_publisher(String, "/dqn/status", 10)

        self.timer = self.create_timer(self.update_period, self.on_timer)
        self.get_logger().info(f"E18 DQN safety trigger started. train_enabled={self.train_enabled}")

    def load_checkpoint(self):
        if not self.model_path.exists():
            return
        try:
            ckpt = self.torch.load(str(self.model_path), map_location="cpu")
            self.q.load_state_dict(ckpt["q"])
            self.target_q.load_state_dict(ckpt["target_q"])
            self.opt.load_state_dict(ckpt["opt"])
            self.epsilon = float(ckpt.get("epsilon", self.epsilon))
            self.train_steps = int(ckpt.get("train_steps", 0))
            self.get_logger().info(f"Loaded DQN checkpoint: {self.model_path}")
        except Exception as exc:
            self.get_logger().warn(f"Failed to load checkpoint: {exc}")

    def save_checkpoint(self):
        self.torch.save({
            "q": self.q.state_dict(),
            "target_q": self.target_q.state_dict(),
            "opt": self.opt.state_dict(),
            "epsilon": self.epsilon,
            "train_steps": self.train_steps,
            "state_keys": STATE_KEYS,
        }, str(self.model_path))

    def on_summary(self, msg):
        try:
            self.latest_summary = json.loads(msg.data)
        except Exception:
            self.latest_summary = None

    def norm_state(self, data):
        goal = float(data.get("goal_distance_m", -1.0))
        obs = float(data.get("min_dynamic_obstacle_distance_m", -1.0))
        safety = float(data.get("safety_scale", 1.0))
        raw = float(data.get("cmd_raw_linear_mps", 0.0))
        safe = float(data.get("cmd_safe_linear_mps", 0.0))
        glen = float(data.get("global_path_length_m", 0.0))
        llen = float(data.get("local_trajectory_length_m", 0.0))

        now = time.time()
        progress = 0.0
        if goal >= 0.0 and self.prev_goal_distance is not None and self.prev_time is not None:
            dt = max(1e-6, now - self.prev_time)
            progress = (self.prev_goal_distance - goal) / dt
        if goal >= 0.0:
            self.prev_goal_distance = goal
            self.prev_time = now

        if obs < 0.0:
            risk = 0.0
            obs_norm = 1.0
        else:
            if obs < 0.20:
                risk = 1.0
            elif obs < 0.50:
                risk = 0.6
            elif obs < 0.80:
                risk = 0.3
            else:
                risk = 0.0
            obs_norm = clamp(obs / 3.0, 0.0, 1.0)

        state_dict = {
            "goal_distance_m": round(goal, 3),
            "min_dynamic_obstacle_distance_m": round(obs, 3),
            "safety_scale": round(safety, 3),
            "raw_linear_mps": round(raw, 3),
            "safe_linear_mps": round(safe, 3),
            "global_path_length_m": round(glen, 3),
            "local_trajectory_length_m": round(llen, 3),
            "progress_mps": round(progress, 3),
            "risk": round(risk, 3),
        }

        vec = [
            clamp(goal / 15.0, 0.0, 1.0) if goal >= 0.0 else 1.0,
            obs_norm,
            clamp(safety, 0.0, 1.0),
            clamp(raw / 0.45, -1.0, 1.0),
            clamp(safe / 0.45, -1.0, 1.0),
            clamp(glen / 25.0, 0.0, 1.0),
            clamp(llen / 6.0, 0.0, 1.0),
            clamp(progress / 0.8, -1.0, 1.0),
            risk,
        ]
        return vec, state_dict

    def is_done(self, data):
        nav = str(data.get("navigation_status", ""))
        goal = float(data.get("goal_distance_m", -1.0))
        return "REACHED" in nav or (goal >= 0.0 and goal < 0.35)

    def reward(self, data, state_dict, action, done):
        obs = float(state_dict["min_dynamic_obstacle_distance_m"])
        safe = float(state_dict["safe_linear_mps"])
        progress = float(state_dict["progress_mps"])
        safety = float(state_dict["safety_scale"])

        r = 0.4 * progress + 0.2 * safe
        r -= 0.25 * max(0.0, 1.0 - safety)

        if obs >= 0.0 and obs < 0.20:
            r -= 2.0
        elif obs >= 0.0 and obs < 0.50:
            r -= 0.5

        if action == 1:
            r -= 0.05

        if abs(safe) < 0.03 and not done:
            r -= 0.05

        if done:
            r += 5.0

        return float(r)

    def select_action(self, vec):
        eps = self.epsilon if self.train_enabled else self.eval_epsilon
        if random.random() < eps:
            return random.randint(0, 1), [0.0, 0.0]

        with self.torch.no_grad():
            qv = self.q(self.torch.tensor([vec], dtype=self.torch.float32))[0]
        return int(self.torch.argmax(qv).item()), [float(qv[0]), float(qv[1])]

    def train_step(self):
        if not self.train_enabled or len(self.replay) < max(self.batch_size, self.warmup_steps):
            return ""

        batch = random.sample(self.replay, self.batch_size)
        s, a, r, ns, d = zip(*batch)

        s = self.torch.tensor(s, dtype=self.torch.float32)
        a = self.torch.tensor(a, dtype=self.torch.long).unsqueeze(1)
        r = self.torch.tensor(r, dtype=self.torch.float32)
        ns = self.torch.tensor(ns, dtype=self.torch.float32)
        d = self.torch.tensor(d, dtype=self.torch.float32)

        q = self.q(s).gather(1, a).squeeze(1)
        with self.torch.no_grad():
            target = r + self.gamma * (1.0 - d) * self.target_q(ns).max(1).values

        loss = self.loss_fn(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_steps == 0:
            self.target_q.load_state_dict(self.q.state_dict())
            self.save_checkpoint()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def publish_adaptation(self, action_id, q_values, state_dict, reward_value):
        action = self.safe_action if action_id == 1 else self.base_action
        source = "dqn_safe_mode" if action_id == 1 else "dqn_baseline"

        adaptation = {
            "schema_version": 1,
            "mode": "e18_dqn_safety_trigger",
            "policy_type": "online_dqn_safety_trigger",
            "policy_source": source,
            "stamp": time.time(),
            "state": state_dict,
            "action": action,
            "dqn": {
                "action_id": action_id,
                "epsilon": round(self.epsilon, 4),
                "q_baseline": round(q_values[0], 4),
                "q_safe": round(q_values[1], 4),
                "train_enabled": self.train_enabled,
            },
            "reward": round(reward_value, 4),
            "speed_scale": action["speed_scale"],
            "lookahead_scale": action["lookahead_scale"],
            "local_horizon_scale": action["local_horizon_scale"],
            "obstacle_caution": action["obstacle_caution"],
        }

        msg = String()
        msg.data = json.dumps(adaptation, separators=(",", ":"))
        self.adaptation_pub.publish(msg)

        smsg = String()
        smsg.data = json.dumps(state_dict, separators=(",", ":"))
        self.state_pub.publish(smsg)

        rmsg = Float32()
        rmsg.data = float(reward_value)
        self.reward_pub.publish(rmsg)

        status = String()
        status.data = f"action={action_id} eps={self.epsilon:.3f} replay={len(self.replay)} train_steps={self.train_steps}"
        self.status_pub.publish(status)

    def on_timer(self):
        if self.latest_summary is None:
            return

        vec, state_dict = self.norm_state(self.latest_summary)
        done = self.is_done(self.latest_summary)

        if self.done_latched:
            if done:
                self.publish_adaptation(0, [0.0, 0.0], state_dict, 0.0)
                return
            self.done_latched = False
            self.prev_state_vec = None
            self.prev_action = None

        reward_value = 0.0
        loss = ""

        if self.prev_state_vec is not None and self.prev_action is not None:
            reward_value = self.reward(self.latest_summary, state_dict, self.prev_action, done)
            if self.train_enabled:
                self.replay.append((self.prev_state_vec, self.prev_action, reward_value, vec, float(done)))
                loss = self.train_step()

        action_id, q_values = self.select_action(vec)
        self.publish_adaptation(action_id, q_values, state_dict, reward_value)

        self.log_writer.writerow({
            "time_s": round(time.time(), 3),
            "episode_id": self.episode_id,
            "step": self.total_steps,
            "action": action_id,
            "reward": round(reward_value, 4),
            "done": int(done),
            "goal_distance_m": state_dict["goal_distance_m"],
            "min_obstacle_m": state_dict["min_dynamic_obstacle_distance_m"],
            "progress_mps": state_dict["progress_mps"],
            "epsilon": round(self.epsilon, 4),
            "loss": loss,
            "q_baseline": round(q_values[0], 4),
            "q_safe": round(q_values[1], 4),
        })
        self.log_file.flush()

        self.total_steps += 1

        if done and not self.done_latched:
            self.done_latched = True
            self.save_checkpoint()
            self.episode_id += 1
            self.prev_state_vec = None
            self.prev_action = None
        else:
            self.prev_state_vec = vec
            self.prev_action = action_id

    def destroy_node(self):
        try:
            self.save_checkpoint()
            self.log_file.flush()
            self.log_file.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = DqnSafetyTrigger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except rclpy.executors.ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
