#!/bin/bash
# 錄影供選單 viewer：三果 × ckpts 各 N 集，存 3cam mp4 + EEF csv（給 eval_menu_viewer.py）。
# 每支 recording 間 strong cleanup 防 Isaac degrade。用法: bash scripts/tools/record_v2_menu.sh [N=10] [ckpts...]
set -u
PROJ=/home/teddyminelab/openarm_isaaclab
cd "$PROJ" || exit 1
N=${1:-10}
shift || true
CKPTS="${*:-060000 070000 080000 090000 100000}"
echo "[recview-all] ===== 三果 v2 錄影 N=$N ckpts={$CKPTS} start=$(date '+%m-%d %H:%M') ====="
for fr in apple strawberry banana; do
  ds=handover_${fr}_v2
  for ckpt in $CKPTS; do
    echo "[recview-all] >>>>> $fr $ckpt start $(date '+%m-%d %H:%M') <<<<<"
    bash scripts/tools/eval_record_view_act.sh "$fr" "$ds" "$ckpt" "$N" \
        > outputs/recview_${fr}_v2_${ckpt}.log 2>&1
    dn=$(grep -aE '\[DONE\]' outputs/recview_${fr}_v2_${ckpt}.log | tail -1)
    echo "[recview-all] <<<<< $fr $ckpt DONE :: ${dn:-NO RESULT} $(date '+%H:%M') >>>>>"
    pkill -9 -f 'act_bridge_node' 2>/dev/null
    pkill -9 -f 'run_handover_eval_scen' 2>/dev/null
    pkill -9 -f 'isaaclab.sh' 2>/dev/null
    pkill -9 -f 'record_3cam' 2>/dev/null
    pkill -9 -f '_ros2_daemon' 2>/dev/null
    rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null
    sleep 15
  done
done
echo "[recview-all] ===== ALL RECORD DONE $(date '+%m-%d %H:%M') ====="