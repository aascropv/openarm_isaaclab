#!/usr/bin/env bash
# 在 lerobot conda env 中啟動 SmolVLA + RTC bridge — 8D 純左臂版（v3 finetune）
set -e

source /opt/ros/jazzy/setup.bash
source "$HOME/openarm_isaaclab/ros2_ws/install/setup.bash"

exec conda run -n lerobot --no-capture-output \
    python3 "$HOME/openarm_isaaclab/policy_bridges/smolvla_rtc_bridge_left8d_node.py" "$@"
