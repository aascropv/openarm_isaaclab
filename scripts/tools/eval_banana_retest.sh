#!/bin/bash
# banana 三 ckpt 乾淨重測：排除 sweep 累積 degrade 污染（banana 對 Isaac/ROS/DDS degrade 最敏感）。
# 每個 ckpt 前做「強清理」（殺 GPU 進程 + ros2 daemon stop + 清所有 dds shm + settle），拿乾淨數字。
# 用法: bash scripts/tools/eval_banana_retest.sh [N] [EH]
set -u
PROJ=/home/teddyminelab/openarm_isaaclab
cd "$PROJ" || exit 1
N=${1:-10}
EH=${2:-25}
DATASET=handover_multitask_v2
SUM=outputs/eval_banana_retest_summary.txt

strong_cleanup() {
  pkill -9 -f 'smolvla_rtc_bridge_node' 2>/dev/null
  pkill -9 -f 'run_handover_eval_scene' 2>/dev/null
  pkill -9 -f 'isaaclab.sh' 2>/dev/null
  pkill -9 -f 'benchmark_handover' 2>/dev/null
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
  ros2 daemon stop >/dev/null 2>&1
  rm -f /dev/shm/fastrtps_* /dev/shm/fast_* /dev/shm/sem.* /dev/shm/*ros* 2>/dev/null
  rm -f /tmp/openarm_episode.json /tmp/eval_scene.log 2>/dev/null
  sleep 12
}

: > "$SUM"
echo "[retest] banana N=$N EH=$EH ckpts={30k 40k 50k}, 每 ckpt 前強清理  start=$(date '+%m-%d %H:%M')" | tee -a "$SUM"
for ckpt in 030000 040000 050000; do
  echo "[retest] --- strong cleanup before $ckpt ---" | tee -a "$SUM"
  strong_cleanup
  logf=outputs/eval_v2_${ckpt}_banana_retest.log
  echo "[retest] >>> banana ckpt=$ckpt ($(date '+%H:%M'))  log=$logf" | tee -a "$SUM"
  bash scripts/tools/eval_fruit.sh banana "$DATASET" "$ckpt" "$N" "$EH" > "$logf" 2>&1
  dl=$(grep -aE "\[DONE\] " "$logf" | tail -1)
  echo "[retest][RESULT] banana ckpt=$ckpt :: ${dl:-NO RESULT (see $logf)}" | tee -a "$SUM"
done
strong_cleanup
echo "[retest] ALL DONE  end=$(date '+%m-%d %H:%M')" | tee -a "$SUM"
