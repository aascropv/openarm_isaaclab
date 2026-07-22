#!/usr/bin/env bash
# 啟動 ROS2 joint + cube trajectory logger（避開 conda env Python 版本衝突）
# Usage: bash scripts/tools/run_log_joint_trajectory.sh --output_dir results_xxx --filename traj.csv
set -e

source /opt/ros/jazzy/setup.bash
source "$HOME/openarm_isaaclab/ros2_ws/install/setup.bash"

# 用 system python3（Ubuntu 24.04 = Python 3.12，跟 ROS Jazzy bindings 匹配）
exec /usr/bin/python3 "$HOME/openarm_isaaclab/scripts/tools/log_joint_trajectory.py" "$@"
