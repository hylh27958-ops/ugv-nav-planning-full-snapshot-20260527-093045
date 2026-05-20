from pathlib import Path

import pandas as pd


DATA_DIR = Path("/root/ugv_nav_ws/results/c15_multi_map_rl_dataset")
OUT_SUMMARY = DATA_DIR / "c15_dataset_summary.csv"
OUT_VALID = DATA_DIR / "c15_all_valid_transitions.csv"


def valid_rows(df):
    return df[
        (df["goal_distance_m"] >= 0.0)
        & (df["min_dynamic_obstacle_distance_m"] >= 0.0)
        & (df["navigation_status"] != "UNKNOWN")
    ].copy()


def summarize_file(path):
    df = pd.read_csv(path)
    valid = valid_rows(df)

    return {
        "file": path.name,
        "raw_rows": len(df),
        "valid_rows": len(valid),
        "episodes": int(valid["episode_id"].nunique()) if len(valid) else 0,
        "done_samples": int(valid["done"].sum()) if len(valid) else 0,
        "reached_samples": int(valid["reached"].sum()) if len(valid) else 0,
        "mean_reward": round(float(valid["reward"].mean()), 4) if len(valid) else 0.0,
        "min_obstacle_valid_m": round(float(valid["min_dynamic_obstacle_distance_m"].min()), 3) if len(valid) else -1.0,
        "mean_obstacle_valid_m": round(float(valid["min_dynamic_obstacle_distance_m"].mean()), 3) if len(valid) else -1.0,
        "mean_speed_scale": round(float(valid["speed_scale"].mean()), 3) if len(valid) else 0.0,
        "mean_lookahead_scale": round(float(valid["lookahead_scale"].mean()), 3) if len(valid) else 0.0,
        "mean_local_horizon_scale": round(float(valid["local_horizon_scale"].mean()), 3) if len(valid) else 0.0,
        "mean_obstacle_caution": round(float(valid["obstacle_caution"].mean()), 3) if len(valid) else 0.0,
    }, valid


def main():
    paths = sorted(DATA_DIR.glob("c15_seed*_transitions.csv"))

    if not paths:
        raise SystemExit(f"No C1.5 transition CSV files found in {DATA_DIR}")

    summaries = []
    valid_frames = []

    for path in paths:
        summary, valid = summarize_file(path)
        summaries.append(summary)

        if len(valid):
            valid = valid.copy()
            valid["source_file"] = path.name
            valid_frames.append(valid)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    if valid_frames:
        all_valid = pd.concat(valid_frames, ignore_index=True)
    else:
        all_valid = pd.DataFrame()

    all_valid.to_csv(OUT_VALID, index=False)

    print(summary_df.to_csv(index=False))
    print(f"Saved summary: {OUT_SUMMARY}")
    print(f"Saved valid dataset: {OUT_VALID}")
    print(f"Total valid rows: {len(all_valid)}")
    if len(all_valid):
        print(f"Total episodes: {all_valid['episode_id'].nunique()} per file not globally unique")
        print(f"Done samples: {int(all_valid['done'].sum())}")
        print(f"Reached samples: {int(all_valid['reached'].sum())}")


if __name__ == "__main__":
    main()
