#!/bin/bash
# ACT 單果 ckpt sweep：序列跑多個 checkpoint（單 GPU），每個 ckpt 間徹底清理避免 Isaac degrade 污染。
# 用法: bash scripts/tools/eval_sweep_act.sh <fruit> <dataset> <N> [ckpt1 ckpt2 ...]
# 例: bash scripts/tools/eval_sweep_act.sh apple handover_apple 10 020000 030000 040000 050000 060000
set -u
PROJ=/home/teddyminelab/openarm_isaaclab
cd "$PROJ" || exit 1
FRUIT=${1:?用法: eval_sweep_act.sh <fruit> <dataset> <N> [ckpts...]}
DATASET=${2:?缺 dataset}
N=${3:-10}
shift 3
CKPTS="${*:-020000 030000 040000 050000 060000}"
SUMMARY=outputs/eval_sweep_act_${DATASET}_summary.txt

strong_cleanup() {
  pkill -9 -f 'act_bridge_node' 2>/dev/null
  pkill -9 -f 'run_handover_eval_scen' 2>/dev/null
  pkill -9 -f 'isaaclab.sh' 2>/dev/null
  pkill -9 -f 'benchmark_handover' 2>/dev/null
  pkill -9 -f 'kit' 2>/dev/null
  pkill -9 -f '_ros2_daemon' 2>/dev/null   # ★ 直接殺 daemon，不 source ROS（set -u 下 source 會炸）
  rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null
  sleep 15
}

: > "$SUMMARY"
echo "[sweep-act] fruit=$FRUIT dataset=$DATASET N=$N ckpts={$CKPTS} start=$(date '+%m-%d %H:%M')" | tee -a "$SUMMARY"
for ckpt in $CKPTS; do
  logf=outputs/eval_act_${DATASET}_${ckpt}_${FRUIT}.log
  echo "[sweep-act] >>> ckpt=$ckpt  ($(date '+%H:%M'))  log=$logf" | tee -a "$SUMMARY"
  bash scripts/tools/eval_act_fruit.sh "$FRUIT" "$DATASET" "$ckpt" "$N" > "$logf" 2>&1
  done_line=$(grep -aE "\[DONE\] " "$logf" | tail -1)
  echo "[sweep-act][RESULT] ckpt=$ckpt :: ${done_line:-NO RESULT (see $logf)}" | tee -a "$SUMMARY"
  echo "[sweep-act] strong cleanup before next ckpt..." | tee -a "$SUMMARY"
  strong_cleanup
done
echo "[sweep-act] ALL DONE  end=$(date '+%m-%d %H:%M')" | tee -a "$SUMMARY"
strong_cleanup
echo "[sweep-act] cleaned up. summary → $SUMMARY"
