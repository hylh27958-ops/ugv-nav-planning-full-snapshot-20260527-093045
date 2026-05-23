# E17 RL Safety-Trigger Experiment Report

## Motivation

E13 showed that the original always-on Torch SAC parameter policy did not generalize well on unseen seeds 101-110. E17 reframes reinforcement learning as a safety-assist trigger instead of a continuous parameter controller.

## E17 Method

E17 keeps baseline Pure Pursuit parameters by default. SAC only activates a fixed safe mode when obstacle distance is below 0.80 m and the SAC actor indicates low speed or high caution intent.

Safe mode uses speed_scale=0.85, lookahead_scale=1.10, local_horizon_scale=1.20, obstacle_caution=1.35.

## 10-Seed Comparison on Seeds 101-110

| method             |   runs |   success_count |   mean_duration_s |   mean_safe_linear_mps |   mean_min_obstacle_m |   total_danger_lt_0p20 |   danger_rate_per_s |   total_caution_lt_0p50 |   caution_rate_per_s |   mean_safety_scale |   mean_stop_like_samples |
|:-------------------|-------:|----------------:|------------------:|-----------------------:|----------------------:|-----------------------:|--------------------:|------------------------:|---------------------:|--------------------:|-------------------------:|
| baseline_pp        |     10 |              10 |           164.46  |                 0.379  |                0.157  |                     71 |           0.0431716 |                     715 |             0.434756 |              0.886  |                     36.2 |
| c3_torch_sac_pp    |     10 |              10 |           194.72  |                 0.319  |                0.123  |                    111 |           0.0570049 |                     913 |             0.468878 |              0.872  |                     45.5 |
| e17_safety_trigger |     10 |              10 |           173.992 |                 0.3538 |                0.1048 |                     79 |           0.0454044 |                     555 |             0.31898  |              0.8476 |                     71.3 |


## E17 Per-Seed Detail

| method             |   seed |   success |   duration_s |   final_goal_distance_m |   mean_safe_linear_mps |   min_obstacle_m |   danger_lt_0p20 |   caution_lt_0p50 |   mean_safety_scale |   stop_like_samples |
|:-------------------|-------:|----------:|-------------:|------------------------:|-----------------------:|-----------------:|-----------------:|------------------:|--------------------:|--------------------:|
| e17_safety_trigger |    101 |         1 |       165.8  |                   0.227 |                  0.341 |            0.261 |                0 |                38 |               0.822 |                  86 |
| e17_safety_trigger |    102 |         1 |       179.8  |                   0.221 |                  0.344 |            0.01  |               10 |                65 |               0.824 |                  69 |
| e17_safety_trigger |    103 |         1 |       154.72 |                   0.22  |                  0.372 |            0.096 |                7 |                33 |               0.889 |                  59 |
| e17_safety_trigger |    104 |         1 |       175.8  |                   0.222 |                  0.343 |            0.108 |                6 |                50 |               0.819 |                  61 |
| e17_safety_trigger |    105 |         1 |       169.8  |                   0.22  |                  0.376 |           -0     |               13 |                32 |               0.896 |                  50 |
| e17_safety_trigger |    106 |         1 |       164.8  |                   0.216 |                  0.341 |            0.102 |                6 |                58 |               0.826 |                  97 |
| e17_safety_trigger |    107 |         1 |       165.8  |                   0.209 |                  0.365 |            0.117 |                4 |                63 |               0.877 |                  73 |
| e17_safety_trigger |    108 |         1 |       191.8  |                   0.219 |                  0.34  |            0.151 |                8 |                80 |               0.814 |                  92 |
| e17_safety_trigger |    109 |         1 |       185.8  |                   0.22  |                  0.364 |            0.078 |                9 |                72 |               0.866 |                  55 |
| e17_safety_trigger |    110 |         1 |       185.8  |                   0.22  |                  0.352 |            0.125 |               16 |                64 |               0.843 |                  71 |


## RL Trigger Diagnosis

|   seed |   rows |   trigger_active |   real_risk |   rl_intent |   trigger_rate |   risk_trigger_coverage |   min_obs_all |   min_obs_when_active |   mean_torch_speed |   mean_torch_caution |
|-------:|-------:|-----------------:|------------:|------------:|---------------:|------------------------:|--------------:|----------------------:|-------------------:|---------------------:|
|    101 |    541 |               99 |          99 |         325 |          0.183 |                       1 |         0.261 |                 0.261 |              0.869 |                1.366 |
|    102 |    801 |              106 |         106 |         454 |          0.132 |                       1 |        -0.038 |                 0.01  |              0.861 |                1.378 |
|    103 |    109 |                3 |           3 |          17 |          0.028 |                       1 |         0.79  |                 0.79  |              0.948 |                1.182 |
|    104 |    358 |               20 |          20 |         131 |          0.056 |                       1 |         0.351 |                 0.351 |              0.903 |                1.296 |
|    105 |    742 |              121 |         121 |         381 |          0.163 |                       1 |        -0.04  |                 0.061 |              0.88  |                1.341 |
|    106 |     86 |               13 |          13 |          22 |          0.151 |                       1 |         0.46  |                 0.46  |              0.92  |                1.231 |
|    107 |    703 |              108 |         108 |         352 |          0.154 |                       1 |         0.117 |                 0.117 |              0.88  |                1.334 |
|    108 |    835 |              146 |         146 |         477 |          0.175 |                       1 |         0.151 |                 0.151 |              0.86  |                1.376 |
|    109 |    820 |              136 |         136 |         406 |          0.166 |                       1 |         0.078 |                 0.078 |              0.877 |                1.341 |
|    110 |    805 |              131 |         131 |         427 |          0.163 |                       1 |         0.125 |                 0.125 |              0.872 |                1.357 |


## E17.1 Stronger Trigger Rejection

E17.1 was tested on seeds 102, 105, and 110 with stronger safe-mode parameters. It is rejected because it increased dangerous near-obstacle exposure on seeds 102 and 105.

| method             |   seed |   success |   duration_s |   mean_safe_linear_mps |   min_obstacle_m |   danger_lt_0p20 |   caution_lt_0p50 |   mean_safety_scale |
|:-------------------|-------:|----------:|-------------:|-----------------------:|-----------------:|-----------------:|------------------:|--------------------:|
| e171_safer_trigger |    102 |         1 |       185.8  |                  0.336 |            0.012 |               17 |                65 |               0.834 |
| e171_safer_trigger |    105 |         1 |       185.8  |                  0.341 |            0.006 |               37 |               124 |               0.833 |
| e171_safer_trigger |    110 |         1 |       184.89 |                  0.349 |            0.238 |                0 |                41 |               0.854 |


## Conclusion

E17 substantially improves over original C3 SAC on unseen seeds 101-110, reducing duration, increasing safe speed, and lowering near-obstacle exposure. However, it still does not beat the strong baseline PP + Safety Filter.

Paper framing: RL safety triggering improves the generalization of the learned layer compared with always-on SAC, while preserving most nominal controller behavior. The strong classical baseline remains slightly better on the current 10 unseen maps.
