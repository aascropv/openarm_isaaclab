#!/bin/bash
# 三果 ACT 40k N=20 錄製複測：逐果跑 eval_record_view_act.sh，果間 strong_cleanup。
# 每果產出 outputs/evalview_act_handover_<fruit>_<ckpt>.csv + _3cam.mp4，供 eval_viewer.py 檢視。
# 用法: bash scripts/tools/eval_recordview_all_act.sh [ckpt] [N]
set -u
PROJ=/home/teddyminelab/openarm_isaaclab
cd "$PROJ" || exit 1
CKPT=${1:-040000}
N=${2:-20}
SUMMARY=outputs/eval_recordview_all_act_summary.txt

strong_cleanup() {
  pkill -9 -f 'act_bridge_node' 2>/dev/null
  pkill -9 -f 'run_handover_eval_scen' 2>/dev/null
  pkill -9 -f 'isaaclab.sh' 2>/dev/null
  pkill -9 -f 'benchmark_handover' 2>/dev/null
  pkill -9 -f 'record_3cam' 2>/dev/null
  pkill -9 -f 'kit' 2>/dev/null
  pkill -9 -f '_ros2_daemon' 2>/dev/null
  rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null
  sleep 15
}

: > "$SUMMARY"
echo "[recview-all] ckpt=$CKPT N=$N start=$(date '+%m-%d %H:%M')" | tee -a "$SUMMARY"
for fruit in apple strawberry banana; do
  ds=handover_${fruit}
  logf=outputs/recview_act_${fruit}_${CKPT}.log
  echo "[recview-all] >>> $fruit ($(date '+%H:%M')) log=$logf" | tee -a "$SUMMARY"
  bash scripts/tools/eval_record_view_act.sh "$fruit" "$ds" "$CKPT" "$N" > "$logf" 2>&1
  res=$(grep -aE '\[DONE\]' "$logf" | tail -1)
  base=outputs/evalview_act_${ds}_${CKPT}
  echo "[recview-all][RESULT] $fruit :: ${res:-NO RESULT} :: base=$base" | tee -a "$SUMMARY"
  strong_cleanup
done
echo "[recview-all] ALL DONE end=$(date '+%m-%d %H:%M')" | tee -a "$SUMMARY"
echo "[recview-all] 檢視: conda run -n lerobot python3 scripts/tools/eval_viewer.py --base outputs/evalview_act_handover_<fruit>_${CKPT} --fruit <fruit> --basket <red|yellow>" | tee -a "$SUMMARY"
