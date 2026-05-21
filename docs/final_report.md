# UGV Path Planning Experiment Final Report

## Project Objective

This project implements a ROS2-based UGV navigation system using A* global planning, Pure Pursuit path tracking, SAC adaptive parameter policy, and Safety Filter velocity constraint.

Final pipeline:

A* Global Planner -> Pure Pursuit -> Torch SAC Adaptive Policy -> Safety Filter -> Robot Motion -> Metrics Recorder

## SAC Policy

The SAC policy is an offline SAC / batch RL policy. It does not directly control robot velocity. Instead, it adapts speed_scale, lookahead_scale, local_horizon_scale, and obstacle_caution.

## Compared Methods

baseline_pp
c3_torch_sac_pp
c44_risk_v2_torch_sac_pp
c47_risk_v2_safety_gate_pp

Each method was tested on five random seeds: 8, 12, 21, 34, 55.

## Final Results

baseline_pp: success 5/5, duration 159.52 s, safe speed 0.362 m/s, danger_lt_0p20 46, score 18.84
c3_torch_sac_pp: success 5/5, duration 171.96 s, safe speed 0.332 m/s, danger_lt_0p20 20, score 108.75
c44_risk_v2_torch_sac_pp: success 5/5, duration 153.08 s, safe speed 0.394 m/s, danger_lt_0p20 40, score 38.87
c47_risk_v2_safety_gate_pp: success 5/5, duration 153.04 s, safe speed 0.387 m/s, danger_lt_0p20 33, score 53.35

## Final Decision

The final safety-prioritized method is A* + Torch SAC Adaptive Pure Pursuit + Safety Filter, corresponding to c3_torch_sac_pp.

C3 is selected because it achieves the highest risk-weighted score and the lowest dangerous near-obstacle exposure.

C4.4 is retained as the efficiency-prioritized comparison method. C4.7 Safety Gate is retained as an ablation experiment.

## Current Level

This work satisfies the complete engineering requirement of A* + PP + SAC + Safety Filter. It is a complete experimental prototype, but not yet a full publication-level study.

Future paper-level work should include 30-50 random maps, stronger baselines such as DWA/TEB/MPC, train-test map separation, Gazebo or real-robot validation, and statistical analysis.
