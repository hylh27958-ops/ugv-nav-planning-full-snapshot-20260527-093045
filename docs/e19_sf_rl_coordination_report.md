# E19 Safety-Filter-Aware RL Coordination Report

## Motivation

E19 tested whether the RL adaptation layer and the deterministic Safety Filter can be coordinated instead of replacing the Safety Filter.

## E17 Conflict Diagnosis

E17 trigger diagnostics showed that most RL trigger activations occurred while the Safety Filter was still clear.

|   seed |   rows |   trigger_rate |   active_samples |   active_hard_filter |   active_warning_filter |   active_clear_filter |   active_obs_far |   active_obs_warning |   active_obs_danger |   hard_filter_samples |   warning_filter_samples |   clear_filter_samples |   min_obs_all |   mean_safety_scale |   mean_safe_speed | file                                                                                            |
|-------:|-------:|---------------:|-----------------:|---------------------:|------------------------:|----------------------:|-----------------:|---------------------:|--------------------:|----------------------:|-------------------------:|-----------------------:|--------------:|--------------------:|------------------:|:------------------------------------------------------------------------------------------------|
|    101 |    541 |          0.183 |               99 |                   15 |                      37 |                    47 |                0 |                   99 |                   0 |                    41 |                      120 |                    380 |         0.261 |               0.86  |             0.352 | /root/ugv_nav_ws/results/stage_e17_safety_trigger/rl_seed101/rl_transitions_20260523_155302.csv |
|    102 |    801 |          0.132 |              106 |                   16 |                       5 |                    85 |                0 |                   96 |                  10 |                    39 |                      159 |                    603 |         0.01  |               0.867 |             0.362 | /root/ugv_nav_ws/results/stage_e17_safety_trigger/rl_seed102/rl_transitions_20260523_155555.csv |
|    103 |    109 |          0.028 |                3 |                    0 |                       0 |                     3 |                0 |                    3 |                   0 |                     1 |                        0 |                    108 |         0.79  |               0.991 |             0.423 | /root/ugv_nav_ws/results/stage_e17_safety_trigger/rl_seed103/rl_transitions_20260523_155903.csv |
|    104 |    358 |          0.056 |               20 |                    0 |                       0 |                    20 |                0 |                   20 |                   0 |                    16 |                       40 |                    302 |         0.351 |               0.913 |             0.391 | /root/ugv_nav_ws/results/stage_e17_safety_trigger/rl_seed104/rl_transitions_20260523_160146.csv |
|    105 |    742 |          0.163 |              121 |                    0 |                      28 |                    93 |                0 |                  108 |                  13 |                    18 |                       77 |                    647 |         0.061 |               0.945 |             0.395 | /root/ugv_nav_ws/results/stage_e17_safety_trigger/rl_seed105/rl_transitions_20260523_160449.csv |
|    106 |     86 |          0.151 |               13 |                    0 |                       1 |                    12 |                0 |                   13 |                   0 |                     3 |                        1 |                     82 |         0.46  |               0.963 |             0.41  | /root/ugv_nav_ws/results/stage_e17_safety_trigger/rl_seed106/rl_transitions_20260523_160748.csv |
|    107 |    703 |          0.154 |              108 |                    0 |                       6 |                   102 |                0 |                  104 |                   4 |                    22 |                       39 |                    642 |         0.117 |               0.954 |             0.396 | /root/ugv_nav_ws/results/stage_e17_safety_trigger/rl_seed107/rl_transitions_20260523_161042.csv |
|    108 |    835 |          0.175 |              146 |                   17 |                      32 |                    97 |                0 |                  138 |                   8 |                    64 |                      126 |                    645 |         0.151 |               0.874 |             0.365 | /root/ugv_nav_ws/results/stage_e17_safety_trigger/rl_seed108/rl_transitions_20260523_161336.csv |
|    109 |    820 |          0.166 |              136 |                   19 |                      45 |                    72 |                0 |                  127 |                   9 |                    44 |                       89 |                    687 |         0.078 |               0.911 |             0.383 | /root/ugv_nav_ws/results/stage_e17_safety_trigger/rl_seed109/rl_transitions_20260523_161654.csv |
|    110 |    805 |          0.163 |              131 |                   10 |                      45 |                    76 |                0 |                  115 |                  16 |                    51 |                      108 |                    646 |         0.125 |               0.897 |             0.375 | /root/ugv_nav_ws/results/stage_e17_safety_trigger/rl_seed110/rl_transitions_20260523_162007.csv |

## E19 Full Parameter Coordination

| method          |   seed |   success | final_status   |   duration_s |   final_goal_distance_m |   mean_safe_linear_mps |   min_obstacle_m |   danger_lt_0p20 |   caution_lt_0p50 |   mean_safety_scale |   stop_like_samples |
|:----------------|-------:|----------:|:---------------|-------------:|------------------------:|-----------------------:|-----------------:|-----------------:|------------------:|--------------------:|--------------------:|
| e19_sf_aware_rl |    105 |         1 | REACHED        |       195.8  |                    0.21 |                  0.326 |            0.036 |               28 |               150 |               0.817 |                 124 |
| e19_sf_aware_rl |    108 |         0 | SAFETY_STOP    |       300.99 |                   14.75 |                  0.206 |            0.015 |               13 |                31 |               0.529 |                1355 |

## E19.2 Speed-Only Coordination

| method                   |   seed |   success | final_status   |   duration_s |   final_goal_distance_m |   mean_safe_linear_mps |   min_obstacle_m |   danger_lt_0p20 |   caution_lt_0p50 |   mean_safety_scale |   stop_like_samples |
|:-------------------------|-------:|----------:|:---------------|-------------:|------------------------:|-----------------------:|-----------------:|-----------------:|------------------:|--------------------:|--------------------:|
| e192_sf_aware_speed_only |    105 |         0 | NAVIGATING     |       301.1  |                  10.273 |                  0.28  |            0.007 |               71 |               278 |               0.744 |                 957 |
| e192_sf_aware_speed_only |    108 |         0 | SAFETY_STOP    |       301.42 |                  14.182 |                  0.239 |            0.025 |               53 |               218 |               0.62  |                1973 |

## Conclusion

External coordination between the learned trigger and the Safety Filter did not produce a cooperative effect. E19.2 removed lookahead, horizon, and obstacle-caution changes and only allowed speed scaling, but both seed105 and seed108 failed to reach the final goal within the time limit.

This suggests that the current SAC intent signal is not suitable for direct online coordination with the Safety Filter. The interaction creates oscillatory behavior and excessive stop-like samples.

Next step: evaluate Safety Filter removal or replacement only as a simulation ablation, not as the main safety design.