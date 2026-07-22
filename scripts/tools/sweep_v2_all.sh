#!/bin/bash
# 三果 v2 ACT ckpt sweep（單 GPU 序列，每 ckpt 間 strong_cleanup 防 Isaac degrade）。
# 預設 40/50/60k 對齊 v1 比較點。用法: bash scripts/tools/sweep_v2_all.sh [N=10] [ckpts...]
set -u
PROJ=/home/teddyminelab/openarm_isaaclab
cd "$PROJ" || exit 1
N=${1:-10}
shift || true
CKPTS="${*:-040000 050000 060000}"
echo "[sweep-all] ===== 三果 v2 sweep N=$N ckpts={$CKPTS} start=$(date '+%m-%d %H:%M') ====="
for fr in apple strawberry banana; do
  ds=handover_${fr}_v2
  echo "[sweep-all] >>>>> $fr ($ds) start $(date '+%m-%d %H:%M') <<<<<"
  bash scripts/tools/eval_sweep_act.sh "$fr" "$ds" "$N" $CKPTS
  echo "[sweep-all] <<<<< $fr done $(date '+%m-%d %H:%M') >>>>>"
done
echo "[sweep-all] ===== ALL 3 SWEEP DONE $(date '+%m-%d %H:%M') ====="
for fr in apple strawberry banana; do
  echo "  --- handover_${fr}_v2 ---"
  grep -E 'RESULT|ALL DONE' outputs/eval_sweep_act_handover_${fr}_v2_summary.txt 2>/dev/null
done