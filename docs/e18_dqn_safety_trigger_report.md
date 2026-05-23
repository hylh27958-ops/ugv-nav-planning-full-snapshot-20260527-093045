# E18 Online DQN Safety-Trigger Experiment Report

## Motivation

E18 tested whether an online DQN can replace the SAC-derived safety trigger with a discrete trigger policy.

The action space is binary:

- action 0: baseline Pure Pursuit parameters
- action 1: fixed safe mode parameters

## E18 Frozen Evaluation

| method                 |   seed |   success |   duration_s |   final_goal_distance_m |   mean_safe_linear_mps |   min_obstacle_m |   danger_lt_0p20 |   caution_lt_0p50 |   mean_safety_scale |   stop_like_samples |
|:-----------------------|-------:|----------:|-------------:|------------------------:|-----------------------:|-----------------:|-----------------:|------------------:|--------------------:|--------------------:|
| e18_dqn_safety_trigger |    105 |         1 |       185.8  |                   0.211 |                  0.343 |            0.125 |               29 |                97 |               0.856 |                  71 |
| e18_dqn_safety_trigger |    108 |         1 |       199.58 |                   0.226 |                  0.329 |            0.182 |                2 |                61 |               0.817 |                  76 |

## E18 vs E17 on Seeds 105 and 108

|   seed |   success_e17 |   duration_s_e17 |   mean_safe_linear_mps_e17 |   min_obstacle_m_e17 |   danger_lt_0p20_e17 |   caution_lt_0p50_e17 |   mean_safety_scale_e17 |   stop_like_samples_e17 |   success_e18 |   duration_s_e18 |   mean_safe_linear_mps_e18 |   min_obstacle_m_e18 |   danger_lt_0p20_e18 |   caution_lt_0p50_e18 |   mean_safety_scale_e18 |   stop_like_samples_e18 |   delta_success |   delta_duration_s |   delta_mean_safe_linear_mps |   delta_min_obstacle_m |   delta_danger_lt_0p20 |   delta_caution_lt_0p50 |   delta_mean_safety_scale |   delta_stop_like_samples |
|-------:|--------------:|-----------------:|---------------------------:|---------------------:|---------------------:|----------------------:|------------------------:|------------------------:|--------------:|-----------------:|---------------------------:|---------------------:|---------------------:|----------------------:|------------------------:|------------------------:|----------------:|-------------------:|-----------------------------:|-----------------------:|-----------------------:|------------------------:|--------------------------:|--------------------------:|
|    105 |             1 |            169.8 |                      0.376 |               -0     |                   13 |                    32 |                   0.896 |                      50 |             1 |           185.8  |                      0.343 |                0.125 |                   29 |                    97 |                   0.856 |                      71 |               0 |              16    |                       -0.033 |                  0.125 |                     16 |                      65 |                    -0.04  |                        21 |
|    108 |             1 |            191.8 |                      0.34  |                0.151 |                    8 |                    80 |                   0.814 |                      92 |             1 |           199.58 |                      0.329 |                0.182 |                    2 |                    61 |                   0.817 |                      76 |               0 |               7.78 |                       -0.011 |                  0.031 |                     -6 |                     -19 |                     0.003 |                       -16 |

## E18.1 Risk-Gated DQN

E18.1 added a hard risk gate so that DQN safe mode can only activate when the obstacle distance is below 0.80 m.

| method                 |   seed |   success |   duration_s |   final_goal_distance_m |   mean_safe_linear_mps |   min_obstacle_m |   danger_lt_0p20 |   caution_lt_0p50 |   mean_safety_scale |   stop_like_samples |
|:-----------------------|-------:|----------:|-------------:|------------------------:|-----------------------:|-----------------:|-----------------:|------------------:|--------------------:|--------------------:|
| e181_gated_dqn_trigger |    105 |         1 |       183.49 |                   0.217 |                  0.347 |            0.018 |               18 |               101 |               0.863 |                  69 |
| e181_gated_dqn_trigger |    108 |         0 |       297.81 |                   1.076 |                  0.011 |            1.595 |                0 |                 0 |               0.03  |                4149 |

## Conclusion

E18 DQN was able to train and publish trigger decisions, but the frozen policy was unstable. Risk gating reduced the seed105 danger count compared with raw E18, but seed108 failed to reach the goal and spent most of the run in near-stop behavior.

Therefore E18/E18.1 is rejected as the main method. It is useful as a negative ablation showing that naive online DQN safety triggering is less reliable than the E17 SAC-derived rule trigger.

Next direction: E19 should use a supervised or imitation-based binary safety trigger trained from successful E17-style trigger labels, rather than continuing online DQN exploration.