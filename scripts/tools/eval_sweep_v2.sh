#!/bin/bash
# handover_multitask_v2 eval sweep：3 ckpt × 3 水果，序列跑（單 GPU）。
# 每次呼叫 eval_fruit.sh（內含 Isaac scene + bridge 起停 + benchmark），抓 [DONE] 成績彙整。
# 用法: bash scripts/tools/eval_sweep_v2.sh [N] [EH]
set -u
PROJ=/home/teddyminelab/openarm_isaaclab
cd "$PROJ" || exit 1
N=${1:-10}
EH=${2:-25}
DATASET=handover_multitask_v2
CKPTS="030000 040000 050000"
FRUITS="apple strawberry banana"
SUMMARY=outputs/eval_sweep_v2_summary.txt

: > "$SUMMARY"
echo "[sweep] dataset=$DATASET  N=$N  EH=$EH  ckpts={$CKPTS}  fruits={$FRUITS}  start=$(date '+%m-%d %H:%M')" | tee -a "$SUMMARY"
for ckpt in $CKPTS; do
  for fruit in $FRUITS; do
    logf=outputs/eval_v2_${ckpt}_${fruit}.log
    echo "[sweep] >>> ckpt=$ckpt fruit=$fruit  ($(date '+%H:%M'))  log=$logf" | tee -a "$SUMMARY"
    bash scripts/tools/eval_fruit.sh "$fruit" "$DATASET" "$ckpt" "$N" "$EH" > "$logf" 2>&1
    done_line=$(grep -aE "\[DONE\] " "$logf" | tail -1)
    echo "[sweep][RESULT] ckpt=$ckpt fruit=$fruit :: ${done_line:-NO RESULT (see $logf)}" | tee -a "$SUMMARY"
  done
done
echo "[sweep] ALL DONE  end=$(date '+%m-%d %H:%M')" | tee -a "$SUMMARY"
# 收尾清乾淨
pkill -9 -f 'smolvla_rtc_bridge_node' 2>/dev/null
pkill -9 -f 'run_handover_eval_scen' 2>/dev/null
pkill -9 -f 'isaaclab.sh' 2>/dev/null
echo "[sweep] cleaned up"
