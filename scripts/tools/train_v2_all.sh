#!/bin/bash
# 三果 v2 ACT 依序訓練 100k（單 GPU 序列）。用法: bash scripts/tools/train_v2_all.sh [steps]
# 各自輸出 outputs/finetune_act_handover_{fruit}_v2，不動 v1。存每 10k → ckpt 10k..100k。
set -u
PROJ=/home/teddyminelab/openarm_isaaclab
cd "$PROJ" || exit 1
STEPS=${1:-100000}
echo "[train-all] ===== 三果 v2 ACT 訓練 steps=$STEPS start=$(date '+%m-%d %H:%M') ====="
for fr in apple strawberry banana; do
  ds=handover_${fr}_v2
  echo "[train-all] >>>>> $ds start $(date '+%m-%d %H:%M') <<<<<"
  bash scripts/tools/train_act_fruit.sh "$ds" "$STEPS"
  rc=$?
  echo "[train-all] <<<<< $ds done rc=$rc $(date '+%m-%d %H:%M') >>>>>"
  [ $rc -ne 0 ] && { echo "[train-all] $ds FAILED rc=$rc，中止"; exit $rc; }
done
echo "[train-all] ===== ALL 3 TRAIN DONE $(date '+%m-%d %H:%M') ====="