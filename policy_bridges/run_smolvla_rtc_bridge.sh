#!/usr/bin/env bash
# 在 lerobot conda env 中啟動 SmolVLA + RTC bridge node（含 ROS2 Jazzy）
set -e

source /opt/ros/jazzy/setup.bash
source "$HOME/openarm_isaaclab/ros2_ws/install/setup.bash"

exec conda run -n lerobot --no-capture-output \
    python3 "$HOME/openarm_isaaclab/policy_bridges/smolvla_rtc_bridge_node.py" "$@"
