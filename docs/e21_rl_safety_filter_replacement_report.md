# E21 RL Safety-Filter Replacement Report

## Motivation

E21 tests whether the deterministic downstream Safety Filter can be replaced by an SAC-triggered velocity safety layer.

The original safety_filter node is disabled. Pure Pursuit publishes raw velocity commands, and the E21 node publishes the final /cmd_vel by scaling velocity according to SAC safety-trigger signals.

## E21 Selected-Seed Results

| method               |   seed |   ever_reached |   final_success | final_status   |   duration_s |   final_goal_distance_m |   mean_safe_linear_mps |   min_obstacle_m |   danger_lt_0p20 |   caution_lt_0p50 |   mean_safety_scale |   stop_like_samples | file                                          |
|:---------------------|-------:|---------------:|----------------:|:---------------|-------------:|------------------------:|-----------------------:|-----------------:|-----------------:|------------------:|--------------------:|--------------------:|:----------------------------------------------|
| e21_rl_safety_filter |    101 |              1 |               1 | REACHED        |       165.8  |                   0.226 |                  0.38  |            0.058 |                6 |                32 |               0.92  |                  54 | e21_rl_safety_filter_seed101_clean_single.csv |
| e21_rl_safety_filter |    102 |              1 |               1 | REACHED        |       205.8  |                   0.212 |                  0.335 |            0.117 |               10 |                48 |               0.8   |                 193 | e21_rl_safety_filter_seed102_fixed200.csv     |
| e21_rl_safety_filter |    103 |              1 |               1 | REACHED        |       204.12 |                   0.216 |                  0.288 |            0.017 |               13 |                34 |               0.681 |                 318 | e21_rl_safety_filter_seed103_fixed200.csv     |
| e21_rl_safety_filter |    105 |              0 |               0 | NAVIGATING     |       241.1  |                   8.763 |                  0.037 |            0.046 |               60 |               312 |               0.071 |                5018 | e21_rl_safety_filter_seed105.csv              |
| e21_rl_safety_filter |    108 |              0 |               0 | NAVIGATING     |       234.21 |                  10.277 |                  0.101 |            0.003 |               56 |               208 |               0.22  |                5332 | e21_rl_safety_filter_seed108.csv              |

## E17 Reference on Same Seeds

|   seed |   success |   duration_s |   mean_safe_linear_mps |   min_obstacle_m |   danger_lt_0p20 |   caution_lt_0p50 |   mean_safety_scale |   stop_like_samples |
|-------:|----------:|-------------:|-----------------------:|-----------------:|-----------------:|------------------:|--------------------:|--------------------:|
|    101 |         1 |       165.8  |                  0.341 |            0.261 |                0 |                38 |               0.822 |                  86 |
|    102 |         1 |       179.8  |                  0.344 |            0.01  |               10 |                65 |               0.824 |                  69 |
|    103 |         1 |       154.72 |                  0.372 |            0.096 |                7 |                33 |               0.889 |                  59 |
|    105 |         1 |       169.8  |                  0.376 |           -0     |               13 |                32 |               0.896 |                  50 |
|    108 |         1 |       191.8  |                  0.34  |            0.151 |                8 |                80 |               0.814 |                  92 |

## Interpretation

E21 produced strong positive results on selected seeds such as 105 and 108, improving obstacle clearance and reducing danger samples while keeping speed close to the no-safety-filter condition.

However, full-episode tests on seeds 102 and 103 show that the method can become overly conservative. Seed103 in particular had lower speed, more stop-like samples, and worse near-obstacle exposure than E17.

Therefore, RL replacement of the Safety Filter is feasible but not yet robust enough to serve as the main method. It should be treated as a future direction or ablation, while the deterministic Safety Filter remains the safer default design.

## Conclusion

The best current paper framing is that Safety Filter interaction is a key factor in RL navigation performance. Direct replacement by RL can work in selected cases but does not yet generalize reliably across unseen seeds.