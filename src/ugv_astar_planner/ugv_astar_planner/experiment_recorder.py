import csv
import json
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class ExperimentRecorder(Node):
    def __init__(self):
        super().__init__("experiment_recorder")

        self.declare_parameter("output_dir", "~/ugv_nav_ws/results")
        self.declare_parameter("file_prefix", "ugv_metrics")
        self.declare_parameter("flush_every", 10)

        output_dir = self.get_parameter("output_dir").value
        file_prefix = self.get_parameter("file_prefix").value
        self.flush_every = int(self.get_parameter("flush_every").value)

        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"{file_prefix}_{timestamp}.csv")

        self.fieldnames = [
            "time_epoch",
            "elapsed_record_s",
            "navigation_status",
            "safety_status",
            "goal_distance_m",
            "global_path_length_m",
            "local_trajectory_length_m",
            "cmd_raw_linear_mps",
            "cmd_safe_linear_mps",
            "cmd_safe_angular_radps",
            "safety_scale",
            "min_dynamic_obstacle_distance_m",
            "elapsed_since_goal_s",
        ]

        self.file = open(self.csv_path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()

        self.start_time = time.time()
        self.row_count = 0

        self.create_subscription(String, "/metrics/summary", self.on_summary, 10)

        self.get_logger().info(f"Recording metrics to: {self.csv_path}")

    def on_summary(self, msg):
        now = time.time()

        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Failed to parse /metrics/summary JSON.")
            return

        row = {
            "time_epoch": round(now, 3),
            "elapsed_record_s": round(now - self.start_time, 3),
        }

        for key in self.fieldnames:
            if key in row:
                continue
            row[key] = data.get(key, "")

        self.writer.writerow(row)
        self.row_count += 1

        if self.row_count % self.flush_every == 0:
            self.file.flush()

    def destroy_node(self):
        try:
            self.file.flush()
            self.file.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = ExperimentRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()