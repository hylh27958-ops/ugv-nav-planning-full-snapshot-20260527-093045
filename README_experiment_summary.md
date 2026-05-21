# UGV Navigation Experiment Summary

This repository implements a ROS2 UGV navigation experiment using A* + Pure Pursuit + SAC Adaptive Policy + Safety Filter.

## Final Selected Method

A* Global Planner + Torch SAC Adaptive Pure Pursuit + Safety Filter

Experiment name: c3_torch_sac_pp

## Key Result

Across five random map seeds 8, 12, 21, 34, 55, C3 Torch SAC achieved:

success_count = 5 / 5
mean_duration_s = 171.96
mean_safe_linear_mps = 0.332
danger_lt_0p20 = 20
caution_lt_0p50 = 162
weighted_score = 108.75

## Final Ranking

1. c3_torch_sac_pp: score 108.75
2. c47_risk_v2_safety_gate_pp: score 53.35
3. c44_risk_v2_torch_sac_pp: score 38.87
4. baseline_pp: score 18.84

## Conclusion

The project satisfies the complete engineering requirement of A* + PP + SAC + Safety Filter.

The full report is available at docs/final_report.md.
