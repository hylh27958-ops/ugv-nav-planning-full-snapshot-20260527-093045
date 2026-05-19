import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def load_csv(path):
    path = os.path.expanduser(path)
    df = pd.read_csv(path)

    if "elapsed_record_s" not in df.columns:
        raise RuntimeError(f"{path} missing elapsed_record_s")

    return df


def valid_nonnegative(df, column):
    if column not in df.columns:
        return pd.Series(dtype=float)
    return df[df[column] >= 0.0][column]


def count_status_contains(df, keyword):
    if "navigation_status" not in df.columns:
        return 0
    return int(df["navigation_status"].astype(str).str.contains(keyword, na=False).sum())


def summarize(df, method):
    duration = float(df["elapsed_record_s"].iloc[-1] - df["elapsed_record_s"].iloc[0])

    goal_dist = valid_nonnegative(df, "goal_distance_m")
    final_goal_distance = float(goal_dist.iloc[-1]) if len(goal_dist) else None
    min_goal_distance = float(goal_dist.min()) if len(goal_dist) else None

    reached_samples = count_status_contains(df, "REACHED")
    safety_stop_samples = count_status_contains(df, "SAFETY_STOP")

    reached = bool(reached_samples > 0 or (min_goal_distance is not None and min_goal_distance < 0.5))

    safe_vel = valid_nonnegative(df, "cmd_safe_linear_mps")
    raw_vel = valid_nonnegative(df, "cmd_raw_linear_mps")
    safety_scale = valid_nonnegative(df, "safety_scale")
    obs_dist = valid_nonnegative(df, "min_dynamic_obstacle_distance_m")

    if "cmd_safe_linear_mps" in df.columns:
        moving_samples = int((df["cmd_safe_linear_mps"].abs() > 0.03).sum())
    else:
        moving_samples = 0

    safety_limited_samples = int((safety_scale < 0.99).sum()) if len(safety_scale) else 0
    severe_limited_samples = int((safety_scale < 0.5).sum()) if len(safety_scale) else 0
    stop_like_samples = int((safe_vel < 0.03).sum()) if len(safe_vel) else 0

    global_len = valid_nonnegative(df, "global_path_length_m")
    local_len = valid_nonnegative(df, "local_trajectory_length_m")

    return {
        "method": method,
        "duration_s": round(duration, 3),
        "reached": reached,
        "final_goal_distance_m": round(final_goal_distance, 3) if final_goal_distance is not None else "",
        "min_goal_distance_m": round(min_goal_distance, 3) if min_goal_distance is not None else "",
        "mean_raw_linear_mps": round(float(raw_vel.mean()), 3) if len(raw_vel) else "",
        "mean_safe_linear_mps": round(float(safe_vel.mean()), 3) if len(safe_vel) else "",
        "max_safe_linear_mps": round(float(safe_vel.max()), 3) if len(safe_vel) else "",
        "moving_samples": moving_samples,
        "min_dynamic_obstacle_distance_m": round(float(obs_dist.min()), 3) if len(obs_dist) else "",
        "mean_dynamic_obstacle_distance_m": round(float(obs_dist.mean()), 3) if len(obs_dist) else "",
        "mean_safety_scale": round(float(safety_scale.mean()), 3) if len(safety_scale) else "",
        "safety_limited_samples": safety_limited_samples,
        "severe_limited_samples": severe_limited_samples,
        "stop_like_samples": stop_like_samples,
        "safety_stop_samples": safety_stop_samples,
        "mean_global_path_length_m": round(float(global_len.mean()), 3) if len(global_len) else "",
        "mean_local_trajectory_length_m": round(float(local_len.mean()), 3) if len(local_len) else "",
    }


def plot_compare(df_a, label_a, df_b, label_b, out):
    fig, axes = plt.subplots(5, 1, figsize=(13, 16), sharex=False)

    plots = [
        ("goal_distance_m", "Goal distance (m)"),
        ("cmd_safe_linear_mps", "Safe velocity (m/s)"),
        ("safety_scale", "Safety scale"),
        ("min_dynamic_obstacle_distance_m", "Min obstacle distance (m)"),
        ("global_path_length_m", "Global path length (m)"),
    ]

    for ax, (col, ylabel) in zip(axes, plots):
        if col in df_a.columns:
            ax.plot(df_a["elapsed_record_s"], df_a[col], label=label_a)
        if col in df_b.columns:
            ax.plot(df_b["elapsed_record_s"], df_b[col], label=label_b)

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time (s)")
        ax.grid(True)
        ax.legend()

    fig.tight_layout()
    plt.savefig(out, dpi=180)
    print(f"Saved compare plot: {out}")


def plot_bar_summary(summary_df, out):
    numeric_cols = [
        "duration_s",
        "mean_safe_linear_mps",
        "min_dynamic_obstacle_distance_m",
        "mean_safety_scale",
        "safety_limited_samples",
        "severe_limited_samples",
    ]

    available = [c for c in numeric_cols if c in summary_df.columns]

    fig, axes = plt.subplots(len(available), 1, figsize=(10, 3 * len(available)))

    if len(available) == 1:
        axes = [axes]

    for ax, col in zip(axes, available):
        values = pd.to_numeric(summary_df[col], errors="coerce")
        ax.bar(summary_df["method"], values)
        ax.set_title(col)
        ax.grid(axis="y")

    fig.tight_layout()
    plt.savefig(out, dpi=180)
    print(f"Saved summary bar plot: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="First CSV file")
    parser.add_argument("--b", required=True, help="Second CSV file")
    parser.add_argument("--label-a", default="Astar_PurePursuit")
    parser.add_argument("--label-b", default="Astar_TEB")
    parser.add_argument("--out-dir", default="~/ugv_nav_ws/results")
    args = parser.parse_args()

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df_a = load_csv(args.a)
    df_b = load_csv(args.b)

    summary = pd.DataFrame([
        summarize(df_a, args.label_a),
        summarize(df_b, args.label_b),
    ])

    summary_csv = os.path.join(out_dir, "comparison_summary.csv")
    summary_md = os.path.join(out_dir, "comparison_summary.md")
    plot_png = os.path.join(out_dir, "comparison_plot.png")
    bar_png = os.path.join(out_dir, "comparison_summary_bar.png")

    summary.to_csv(summary_csv, index=False)

    with open(summary_md, "w") as f:
        f.write("# Navigation Comparison Summary\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n")

    print("\n=== Comparison Summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved summary CSV: {summary_csv}")
    print(f"Saved summary Markdown: {summary_md}")

    plot_compare(df_a, args.label_a, df_b, args.label_b, plot_png)
    plot_bar_summary(summary, bar_png)


if __name__ == "__main__":
    main()
