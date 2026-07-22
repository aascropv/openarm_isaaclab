#!/usr/bin/env bash
# 啟動 Isaac Sim handover 評測場景（3 水果 2 籃固定位置 + episode reset）。
# 跟 orchestrator (scripts/tools/benchmark_handover.py) 透過 /tmp/openarm_episode.json IPC 同步。
# 跟 bridge (policy_bridges/run_smolvla_rtc_bridge.sh) 透過 ROS2 topic 同步。
set -e

cd "$HOME/openarm_isaaclab"

NOISE_PATTERNS=(
    "No adjacent samples found for interpolation"
    "getSimulationTimeMonotonicAtTime"
    "Render product not valid"
    "DLSS increasing input dimensions"
    "ITimeline callbacks is deprecated"
    "getStringRawInternal: item"
    "Mesh.*update topology"
    "Large simulation step size"
)
NOISE_REGEX=$(IFS='|'; echo "${NOISE_PATTERNS[*]}")

exec conda run -n env_isaaclab --no-capture-output TERM=xterm \
    "$HOME/IsaacLab-2.3.0/isaaclab.sh" \
    -p scripts/isaac/run_handover_eval_scene.py \
    --enable_cameras "$@" \
    2>&1 | grep --line-buffered -vE "$NOISE_REGEX"
