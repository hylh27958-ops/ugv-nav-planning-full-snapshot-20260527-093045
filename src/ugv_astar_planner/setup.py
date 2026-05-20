import os
from glob import glob
from setuptools import find_packages, setup

package_name = "ugv_astar_planner"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="root@example.com",
    description="UGV A* path planning demo.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "path_demo = ugv_astar_planner.path_demo:main",
            "fake_base = ugv_astar_planner.fake_base:main",
            "local_planner = ugv_astar_planner.local_planner:main",
            "pure_pursuit = ugv_astar_planner.pure_pursuit:main",
            "safety_filter = ugv_astar_planner.safety_filter:main",
            "metrics_monitor = ugv_astar_planner.metrics_monitor:main",
            "experiment_recorder = ugv_astar_planner.experiment_recorder:main",
            "scenario_runner = ugv_astar_planner.scenario_runner:main",
            "sac_adapter = ugv_astar_planner.sac_adapter:main",
            "teb_path_sender = ugv_astar_planner.teb_path_sender:main",
            "rl_transition_recorder = ugv_astar_planner.rl_transition_recorder:main",
        ],
    },
)
