#!/bin/bash
# 通用單任務 ACT 訓練。用法: bash scripts/tools/train_act_fruit.sh <dataset_name> [steps]
# 例: bash scripts/tools/train_act_fruit.sh handover_apple 60000
# ★ recipe 照抄已跑通的 basket ACT（outputs/finetune_act_basket，單任務 90%）:
#   chunk_size=60 / n_action_steps=1 / temporal_ensemble_coeff=0.01（temporal ensembling 開）
#   kl_weight=10 / batch 8 / seed 42 / lr 1e-5（由 ACT preset 提供）/ 3 相機 / 16D（從 dataset 自動推）。
# ★ lerobot 預設不同（chunk=100、ensemble=None=關），別用預設。
# ★ ACT 無語言：只能單任務（多任務場景三果同桌，frame-0 輸入相同→會 path-average）。
# ★★ PYTHONUNBUFFERED + --no-capture-output：讓 log 即時寫出，否則 conda run 完整 buffer、monitor 看不到進度。
DATASET=${1:?用法: train_act_fruit.sh <dataset_name> [steps]}
STEPS=${2:-60000}
PROJ=/home/teddyminelab/openarm_isaaclab
cd "$PROJ" || exit 1
ROOT=datasets_lerobot/openarm/$DATASET
OUT=outputs/finetune_act_$DATASET
LOG=outputs/act_${DATASET}_train.log
[ -d "$ROOT" ] || { echo "dataset 不存在: $ROOT（先跑轉換）"; exit 1; }
[ -d "$OUT" ] && { echo "output 已存在: $OUT（先清掉或改名）"; exit 1; }

echo "[train-act] dataset=$ROOT  output=$OUT  steps=$STEPS  batch=8  log=$LOG"
PYTHONUNBUFFERED=1 conda run --no-capture-output -n lerobot lerobot-train \
  --dataset.repo_id="$DATASET" \
  --dataset.root="$ROOT" \
  --policy.type=act \
  --policy.chunk_size=60 \
  --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --policy.kl_weight=10 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=$STEPS \
  --save_freq=10000 \
  --log_freq=200 \
  --num_workers=4 \
  --seed=42 \
  --output_dir="$OUT" \
  --job_name=act \
  > "$LOG" 2>&1
rc=$?
echo "[train-act] exit=$rc (見 $LOG)"
exit $rc
