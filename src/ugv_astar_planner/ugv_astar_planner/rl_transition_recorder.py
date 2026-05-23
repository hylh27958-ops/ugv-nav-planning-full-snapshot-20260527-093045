import csv
import json
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, String


class RlTransitionRecorder(Node):
    def __init__(self):
        super().__init__("rl_transition_recorder")

        self.declare_parameter("output_dir", "~/ugv_nav_ws/results/rl_transitions")
        self.declare_parameter("record_period", 0.2)
        self.declare_parameter("goal_tolerance", 0.35)
        self.declare_parameter("flush_every", 10)

        self.output_dir = os.path.expanduser(str(self.get_parameter("output_dir").value))
        self.record_period = float(self.get_parameter("record_period").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.flush_every = int(self.get_parameter("flush_every").value)

        os.makedirs(self.output_dir, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = os.path.join(self.output_dir, f"rl_transitions_{stamp}.csv")

        self.latest_state = None
        self.latest_adaptation = None
        self.latest_reward = 0.0
        self.latest_summary = {}

        self.episode_id = 0
        self.step = 0
        self.rows_written = 0
        self.done_latched = False
        self.start_time = time.monotonic()
        self.current_goal_key = None

        self.fieldnames = [
            "time_s",
            "episode_id",
            "step",
            "done",
            "reached",
            "navigation_status",
            "safety_status",
            "goal_distance_m",
            "min_dynamic_obstacle_distance_m",
            "safety_scale",
            "raw_linear_mps",
            "safe_linear_mps",
            "global_path_length_m",
            "local_trajectory_length_m",
            "progress_mps",
            "speed_scale",
            "lookahead_scale",
            "local_horizon_scale",
            "obstacle_caution",
            "policy_source",
            "safety_trigger_active",
            "safety_trigger_real_risk",
            "safety_trigger_rl_intent",
            "safety_trigger_reason",
            "torch_speed_scale",
            "torch_obstacle_caution",
            "risk",
            "reward",
        ]

        self.file = open(self.output_path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()

        self.create_subscription(String, "/sac/state", self.on_state, 10)
        self.create_subscription(String, "/sac/adaptation", self.on_adaptation, 10)
        self.create_subscription(Float32, "/sac/reward", self.on_reward, 10)
        self.create_subscription(String, "/metrics/summary", self.on_summary, 10)
        self.create_subscription(PoseStamped, "/astar_goal", self.on_goal, 10)

        self.status_pub = self.create_publisher(String, "/rl/transition_status", 10)

        self.timer = self.create_timer(self.record_period, self.on_timer)

        self.get_logger().info(f"RL transition recorder started: {self.output_path}")

    def parse_json(self, text):
        try:
            return json.loads(text)
        except Exception:
            return {}

    def on_state(self, msg):
        self.latest_state = self.parse_json(msg.data)

    def on_adaptation(self, msg):
        self.latest_adaptation = self.parse_json(msg.data)

    def on_reward(self, msg):
        self.latest_reward = float(msg.data)

    def on_summary(self, msg):
        self.latest_summary = self.parse_json(msg.data)

    def on_goal(self, msg):
        gx = round(msg.pose.position.x, 3)
        gy = round(msg.pose.position.y, 3)
        goal_key = (gx, gy)

        if self.current_goal_key is None:
            self.current_goal_key = goal_key
            return

        if goal_key != self.current_goal_key:
            self.current_goal_key = goal_key
            self.episode_id += 1
            self.step = 0
            self.done_latched = False

    def get_value(self, primary, fallback, key, default=0.0):
        if isinstance(primary, dict) and key in primary:
            return primary[key]
        if isinstance(fallback, dict) and key in fallback:
            return fallback[key]
        return default

    def publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def on_timer(self):
        if self.latest_state is None or self.latest_adaptation is None:
            self.publish_status("WAITING_FOR_SAC_DATA")
            return

        if self.done_latched:
            self.publish_status(
                f"DONE_LATCHED file={self.output_path} rows={self.rows_written}"
            )
            return

        state = self.latest_state
        adaptation = self.latest_adaptation
        action = adaptation.get("action", {})
        trigger = adaptation.get("safety_trigger", {})
        torch_action = trigger.get("torch_action", {})

        goal_distance = float(
            self.get_value(state, self.latest_summary, "goal_distance_m", -1.0)
        )

        navigation_status = str(
            self.latest_summary.get("navigation_status", "UNKNOWN")
        )
        safety_status = str(
            self.latest_summary.get("safety_status", "UNKNOWN")
        )

        reached = (
            navigation_status == "REACHED"
            or (goal_distance >= 0.0 and goal_distance < self.goal_tolerance)
        )

        done = bool(reached)

        row = {
            "time_s": round(time.monotonic() - self.start_time, 3),
            "episode_id": self.episode_id,
            "step": self.step,
            "done": int(done),
            "reached": int(reached),
            "navigation_status": navigation_status,
            "safety_status": safety_status,
            "goal_distance_m": self.get_value(state, self.latest_summary, "goal_distance_m", -1.0),
            "min_dynamic_obstacle_distance_m": self.get_value(state, self.latest_summary, "min_dynamic_obstacle_distance_m", -1.0),
            "safety_scale": self.get_value(state, self.latest_summary, "safety_scale", 0.0),
            "raw_linear_mps": self.get_value(state, self.latest_summary, "raw_linear_mps", 0.0),
            "safe_linear_mps": self.get_value(state, self.latest_summary, "safe_linear_mps", 0.0),
            "global_path_length_m": self.get_value(state, self.latest_summary, "global_path_length_m", 0.0),
            "local_trajectory_length_m": self.get_value(state, self.latest_summary, "local_trajectory_length_m", 0.0),
            "progress_mps": state.get("progress_mps", 0.0),
            "speed_scale": action.get("speed_scale", adaptation.get("speed_scale", 1.0)),
            "lookahead_scale": action.get("lookahead_scale", adaptation.get("lookahead_scale", 1.0)),
            "local_horizon_scale": action.get("local_horizon_scale", adaptation.get("local_horizon_scale", 1.0)),
            "obstacle_caution": action.get("obstacle_caution", adaptation.get("obstacle_caution", 1.0)),
            "policy_source": adaptation.get("policy_source", ""),
            "safety_trigger_active": int(bool(trigger.get("active", False))),
            "safety_trigger_real_risk": int(bool(trigger.get("real_risk", False))),
            "safety_trigger_rl_intent": int(bool(trigger.get("rl_intent", False))),
            "safety_trigger_reason": trigger.get("reason", ""),
            "torch_speed_scale": torch_action.get("speed_scale", ""),
            "torch_obstacle_caution": torch_action.get("obstacle_caution", ""),
            "risk": adaptation.get("risk", 0.0),
            "reward": adaptation.get("reward", self.latest_reward),
        }

        self.writer.writerow(row)
        self.rows_written += 1
        self.step += 1

        if self.rows_written % self.flush_every == 0:
            self.file.flush()

        if done:
            self.done_latched = True
            self.file.flush()

        self.publish_status(
            f"RECORDING file={self.output_path} rows={self.rows_written} episode={self.episode_id}"
        )

    def destroy_node(self):
        try:
            self.file.flush()
            self.file.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = RlTransitionRecorder()
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
