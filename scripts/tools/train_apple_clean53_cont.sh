#!/bin/bash
# apple clean-53 continued finetune：從 apple_v2 40k checkpoint 續訓，只餵「approach 弧乾淨(夠外側+清障+不狂甩)」的 53 筆，
# 目的：把右手 approach 從「模糊平均→常切內角卡平台/黃籃」矯正成一致的外高清路徑。
# ★ 續訓用 --policy.path=<40k ckpt>（該 ckpt config 已是 state[16]/action[16]，不會觸發 base SmolVLA 的 state[6] 截斷 bug）。
#   不加 --policy.type/--load_vlm_weights（權重與 config 都來自 40k ckpt）。
PROJ=/home/teddyminelab/openarm_isaaclab
cd "$PROJ" || exit 1
DATASET=apple_clean53
ROOT=datasets_lerobot/openarm/$DATASET
CKPT=$PROJ/outputs/finetune_smolvla_apple_v2/checkpoints/040000/pretrained_model
OUT=outputs/finetune_smolvla_apple_clean53_cont
LOG=outputs/smolvla_apple_clean53_cont_train.log
[ -d "$ROOT" ] || { echo "dataset 不存在: $ROOT（先跑轉換）"; exit 1; }
[ -d "$CKPT" ] || { echo "起訓 ckpt 不存在: $CKPT"; exit 1; }
[ -d "$OUT" ] && { echo "output 已存在: $OUT（先清掉或改名）"; exit 1; }

echo "[train] continued from $CKPT  dataset=$ROOT  output=$OUT  log=$LOG"
PYTHONUNBUFFERED=1 conda run --no-capture-output -n lerobot lerobot-train \
  --dataset.repo_id="$DATASET" \
  --dataset.root="$ROOT" \
  --policy.path="$CKPT" \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --batch_size=64 \
  --steps=15000 \
  --save_freq=5000 \
  --log_freq=200 \
  --num_workers=4 \
  --seed=1000 \
  --output_dir="$OUT" \
  --job_name=smolvla_clean53 \
  > "$LOG" 2>&1
echo "[train] exit=$? (見 $LOG)"
