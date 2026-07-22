#!/bin/bash
# apple_v2 單任務 SmolVLA 訓練（重收100筆+新相機+trim_lead12 保留回home）。
# 用法: bash scripts/tools/train_apple.sh
# recipe 沿用 v5b=85% baseline：★必須 --policy.type=smolvla + load_vlm_weights，不能用 --policy.path（會把 state 截成 6D）。
PROJ=/home/teddyminelab/openarm_isaaclab
cd "$PROJ" || exit 1
DATASET=apple_v2
ROOT=datasets_lerobot/openarm/$DATASET
OUT=outputs/finetune_smolvla_apple_v2
LOG=outputs/smolvla_apple_v2_train.log
[ -d "$ROOT" ] || { echo "dataset 不存在: $ROOT（先跑轉換）"; exit 1; }
[ -d "$OUT" ] && { echo "output 已存在: $OUT（先清掉或改名）"; exit 1; }

echo "[train] dataset=$ROOT  output=$OUT  log=$LOG"
# PYTHONUNBUFFERED + --no-capture-output：讓 lerobot 的 config/step log 即時寫入 $LOG
# （否則 conda run 會完整 buffer，整個訓練期間 log 都 0 bytes，monitor 看不到進度）
PYTHONUNBUFFERED=1 conda run --no-capture-output -n lerobot lerobot-train \
  --dataset.repo_id="$DATASET" \
  --dataset.root="$ROOT" \
  --policy.type=smolvla \
  --policy.load_vlm_weights=true \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --batch_size=64 \
  --steps=40000 \
  --save_freq=10000 \
  --log_freq=200 \
  --num_workers=4 \
  --seed=1000 \
  --output_dir="$OUT" \
  --job_name=smolvla \
  > "$LOG" 2>&1
echo "[train] exit=$? (見 $LOG)"
