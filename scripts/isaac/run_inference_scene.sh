#!/usr/bin/env bash
# 啟動 Isaac Sim + ROS2 推論場景（grep filter 擋掉已知 noise warnings）。
#
# 用法：
#   bash scripts/isaac/run_inference_scene.sh                # GUI 模式（預設）
#   bash scripts/isaac/run_inference_scene.sh --headless     # headless 模式
#   bash scripts/isaac/run_inference_scene.sh --image_width 640 --image_height 480
set -e

cd "$HOME/openarm_isaaclab"

# 已知會被 Isaac Sim/ROS2 bridge/RTX 持續 spam 的 noise（內容無害，純洗版）
# 若要看完整 log debug：把整段 `2>&1 | grep ...` 拿掉
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
    -p scripts/isaac/run_inference_scene.py \
    --enable_cameras "$@" \
    2>&1 | grep --line-buffered -vE "$NOISE_REGEX"
