from pathlib import Path

CONFIG_DIR = Path("/root/ugv_nav_ws/src/ugv_astar_planner/config")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

scenarios = [
    {
        "name": "seed8",
        "random_seed": 8,
        "random_obstacle_count": 120,
        "dynamic_obstacles": "4.6,1.6,0.20,0.00,3.5,6.0,1.6,1.6;6.5,4.3,0.00,0.18,6.5,6.5,3.0,5.6;9.2,6.7,-0.16,0.00,8.0,10.5,6.7,6.7;10.7,7.5,0.00,-0.14,10.7,10.7,6.0,8.2",
        "goals": "12.0,1.0;1.0,8.0;12.8,8.0;0.8,0.8",
    },
    {
        "name": "seed12",
        "random_seed": 12,
        "random_obstacle_count": 115,
        "dynamic_obstacles": "3.8,2.0,0.18,0.00,2.8,5.8,2.0,2.0;6.9,5.0,0.00,-0.16,6.9,6.9,3.2,6.2;9.8,3.4,-0.14,0.00,8.5,11.0,3.4,3.4;11.0,7.1,0.00,0.13,11.0,11.0,6.0,8.3",
        "goals": "12.3,1.2;1.2,7.8;12.6,7.7;0.8,0.8",
    },
    {
        "name": "seed21",
        "random_seed": 21,
        "random_obstacle_count": 130,
        "dynamic_obstacles": "4.2,6.2,0.16,0.00,3.0,6.2,6.2,6.2;6.2,2.0,0.00,0.17,6.2,6.2,1.4,4.8;8.8,6.8,-0.18,0.00,7.5,10.2,6.8,6.8;10.5,3.8,0.00,-0.14,10.5,10.5,2.5,5.3",
        "goals": "11.8,1.0;1.0,7.5;12.4,8.1;0.8,0.8",
    },
    {
        "name": "seed34",
        "random_seed": 34,
        "random_obstacle_count": 125,
        "dynamic_obstacles": "3.5,3.5,0.17,0.00,2.5,5.5,3.5,3.5;5.8,6.4,0.00,-0.15,5.8,5.8,4.8,7.5;9.5,5.7,-0.15,0.00,8.0,10.8,5.7,5.7;11.3,2.4,0.00,0.14,11.3,11.3,1.4,4.3",
        "goals": "12.5,1.4;1.1,8.1;12.2,7.8;0.8,0.8",
    },
    {
        "name": "seed55",
        "random_seed": 55,
        "random_obstacle_count": 110,
        "dynamic_obstacles": "4.8,2.5,0.15,0.00,3.4,6.2,2.5,2.5;6.6,6.1,0.00,-0.17,6.6,6.6,4.5,7.5;8.9,4.5,-0.16,0.00,7.6,10.6,4.5,4.5;10.8,7.3,0.00,-0.13,10.8,10.8,5.8,8.4",
        "goals": "12.6,1.0;1.0,7.9;12.8,8.0;0.8,0.8",
    },
]

template = """astar_planner_demo:
  ros__parameters:
    width: 140
    height: 90
    resolution: 0.1
    origin_x: 0.0
    origin_y: 0.0
    inflate_radius: 0.18
    start_x: 0.8
    start_y: 0.8
    goal_x: 12.8
    goal_y: 8.0
    timer_period: 0.2
    random_seed: {random_seed}
    random_obstacle_count: {random_obstacle_count}
    block_min_w: 2
    block_max_w: 10
    block_min_h: 2
    block_max_h: 8
    route_corridor_cells: 4
    dynamic_obstacle_radius: 0.35
    dynamic_obstacles: "{dynamic_obstacles}"

fake_base:
  ros__parameters:
    initial_x: 0.8
    initial_y: 0.8
    initial_yaw: 0.0
    update_period: 0.05
    cmd_timeout: 0.5

local_trajectory_planner:
  ros__parameters:
    update_period: 0.1
    horizon_distance: 3.0
    sample_step: 0.15
    obstacle_influence: 0.9
    max_obstacle_shift: 0.35
    smooth_iterations: 3

pure_pursuit:
  ros__parameters:
    update_period: 0.1
    lookahead: 0.7
    max_linear: 0.45
    max_angular: 1.2
    goal_tolerance: 0.25

safety_filter:
  ros__parameters:
    update_period: 0.05
    max_linear: 0.45
    max_angular: 1.2
    max_linear_accel: 0.45
    max_angular_accel: 1.8
    cmd_timeout: 0.5
    front_angle_deg: 60.0
    stop_distance: 0.45
    slow_distance: 1.2

sac_adapter:
  ros__parameters:
    update_period: 0.2
    mode: "b4_v2_proactive_rule_policy"
    caution_distance: 1.40
    warning_distance: 0.75
    emergency_distance: 0.40
    speed_floor: 0.62
    speed_ceiling: 1.0
    lookahead_ceiling: 1.28
    local_horizon_ceiling: 1.45
    obstacle_caution_ceiling: 1.70
    rise_alpha: 0.55
    fall_alpha: 0.22

teb_path_sender:
  ros__parameters:
    action_name: "follow_path"
    controller_id: "FollowPath"
    goal_checker_id: "goal_checker"
    update_period: 0.2
    resend_period: 2.0
    periodic_replan: true
    goal_change_distance: 0.25
    min_path_poses: 2

experiment_recorder:
  ros__parameters:
    output_dir: "~/ugv_nav_ws/results"
    file_prefix: "ugv_metrics"
    flush_every: 10

scenario_runner:
  ros__parameters:
    goals: "{goals}"
    loop: false
    timeout_sec: 70.0
    settle_sec: 2.0
    initial_delay_sec: 3.0
    publish_period_sec: 1.0
"""

for scenario in scenarios:
    path = CONFIG_DIR / f"ugv_demo_c15_{scenario['name']}.yaml"
    path.write_text(template.format(**scenario), encoding="utf-8")
    print(f"created: {path}")
