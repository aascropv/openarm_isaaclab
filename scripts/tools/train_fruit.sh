#!/bin/bash
# 通用單任務/多任務 SmolVLA 訓練。用法: bash scripts/tools/train_fruit.sh <dataset_name>
# 例: bash scripts/tools/train_fruit.sh strawberry_v1
# recipe 沿用 apple_v2=90% baseline：★必須 type=smolvla + load_vlm_weights（不能用 --policy.path）。
# ★★ PYTHONUNBUFFERED + --no-capture-output：讓 log 即時寫出，否則 conda run 完整 buffer、monitor 看不到進度。
DATASET=${1:?用法: train_fruit.sh <dataset_name> [steps]}
STEPS=${2:-40000}
PROJ=/home/teddyminelab/openarm_isaaclab
cd "$PROJ" || exit 1
ROOT=datasets_lerobot/openarm/$DATASET
OUT=outputs/finetune_smolvla_$DATASET
LOG=outputs/smolvla_${DATASET}_train.log
[ -d "$ROOT" ] || { echo "dataset 不存在: $ROOT（先跑轉換）"; exit 1; }
[ -d "$OUT" ] && { echo "output 已存在: $OUT（先清掉或改名）"; exit 1; }

echo "[train] dataset=$ROOT  output=$OUT  steps=$STEPS  log=$LOG"
PYTHONUNBUFFERED=1 conda run --no-capture-output -n lerobot lerobot-train \
  --dataset.repo_id="$DATASET" \
  --dataset.root="$ROOT" \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --batch_size=64 \
  --steps=$STEPS \
  --save_freq=10000 \
  --log_freq=200 \
  --num_workers=4 \
  --seed=1000 \
  --output_dir="$OUT" \
  --job_name=smolvla \
  > "$LOG" 2>&1
echo "[train] exit=$? (見 $LOG)"
