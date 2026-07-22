#!/bin/bash
# 三果 v2 HDF5 → lerobot，統一 trim_motion lead12/trail15（砍每筆頭~3s + 尾~11s 靜止）。
# 輸出 datasets_lerobot/openarm/handover_{fruit}_v2。用法: bash scripts/tools/convert_v2_all.sh
set -u
PROJ=/home/teddyminelab/openarm_isaaclab
cd "$PROJ" || exit 1
LEAD=12; TRAIL=15
echo "[conv] ===== 三果 v2 轉檔 lead=$LEAD trail=$TRAIL start=$(date '+%m-%d %H:%M') ====="
for fr in apple strawberry banana; do
  echo "[conv] >>>>> $fr start $(date '+%H:%M') <<<<<"
  conda run --no-capture-output -n lerobot python3 scripts/tools/hdf5_to_lerobot_dataset.py \
    --hdf5_dir datasets_handover_v2/$fr/*/ \
    --dataset_name handover_${fr}_v2 \
    --trim_motion --trim_lead $LEAD --trim_trail $TRAIL
  rc=$?
  echo "[conv] <<<<< $fr done rc=$rc $(date '+%H:%M') >>>>>"
  [ $rc -ne 0 ] && { echo "[conv] $fr FAILED rc=$rc，中止"; exit $rc; }
done
echo "[conv] ===== ALL 3 CONVERT DONE $(date '+%m-%d %H:%M') ====="
for fr in apple strawberry banana; do
  p=datasets_lerobot/openarm/handover_${fr}_v2/meta/info.json
  [ -f "$p" ] && python3 -c "import json;j=json.load(open('$p'));print('  %s: ep=%d frames=%d (%.0f/ep, %.1fs)'%('handover_${fr}_v2',j['total_episodes'],j['total_frames'],j['total_frames']/j['total_episodes'],j['total_frames']/j['total_episodes']/30))"
done