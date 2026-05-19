import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_if_exists(ax, t, df, column, label, ylabel=None):
    if column not in df.columns:
        ax.text(0.5, 0.5, f"Missing column: {column}", ha="center", va="center")
        ax.grid(True)
        return

    ax.plot(t, df[column], label=label)
    ax.set_ylabel(ylabel or label)
    ax.legend()
    ax.grid(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="Path to metrics CSV file")
    parser.add_argument("--out", default=None, help="Output image path")
    args = parser.parse_args()

    csv_file = os.path.expanduser(args.csv_file)
    df = pd.read_csv(csv_file)

    if "elapsed_record_s" not in df.columns:
        raise RuntimeError("CSV does not contain elapsed_record_s column")

    t = df["elapsed_record_s"]

    fig, axes = plt.subplots(5, 1, figsize=(13, 15), sharex=True)

    plot_if_exists(
        axes[0],
        t,
        df,
        "goal_distance_m",
        "Goal distance",
        "Distance (m)",
    )

    if "cmd_raw_linear_mps" in df.columns:
        axes[1].plot(t, df["cmd_raw_linear_mps"], label="Raw linear cmd")
    if "cmd_safe_linear_mps" in df.columns:
        axes[1].plot(t, df["cmd_safe_linear_mps"], label="Safe linear cmd")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].legend()
    axes[1].grid(True)

    plot_if_exists(
        axes[2],
        t,
        df,
        "safety_scale",
        "Safety scale",
        "Scale",
    )
    axes[2].set_ylim(-0.05, 1.05)

    plot_if_exists(
        axes[3],
        t,
        df,
        "min_dynamic_obstacle_distance_m",
        "Min dynamic obstacle distance",
        "Distance (m)",
    )

    if "global_path_length_m" in df.columns:
        axes[4].plot(t, df["global_path_length_m"], label="Global path length")
    if "local_trajectory_length_m" in df.columns:
        axes[4].plot(t, df["local_trajectory_length_m"], label="Local trajectory length")
    axes[4].set_ylabel("Path length (m)")
    axes[4].set_xlabel("Time (s)")
    axes[4].legend()
    axes[4].grid(True)

    fig.suptitle(os.path.basename(csv_file))
    fig.tight_layout()

    if args.out is None:
        base, _ = os.path.splitext(csv_file)
        out = base + "_plot.png"
    else:
        out = os.path.expanduser(args.out)

    plt.savefig(out, dpi=180)
    print(f"Saved plot to: {out}")


if __name__ == "__main__":
    main()
