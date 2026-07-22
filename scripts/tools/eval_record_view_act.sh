#!/bin/bash
# ACT 版：跑 N 集閉環 eval，同時錄 三相機並排 mp4 + 每幀 wall + EEF frame log(關節/水果/手臂+wall)。
# 供 eval_viewer.py 逐幀檢視。改自 eval_record_view.sh（SmolVLA 版）：換 ACT bridge、無 --task/--eh/--gripper_snap。
# 用法: bash scripts/tools/eval_record_view_act.sh <fruit> <dataset> <ckpt6> <n>
# 例:   bash scripts/tools/eval_record_view_act.sh apple handover_apple 040000 20
# 檢視: conda run -n lerobot python3 scripts/tools/eval_viewer.py \
#         --base outputs/evalview_act_<dataset>_<ckpt> --fruit <fruit> --basket <red|yellow>
FRUIT=${1:?用法: eval_record_view_act.sh <fruit> <dataset> <ckpt> <n>}
DATASET=${2:?缺 dataset}
CKPT=${3:-040000}
NEPS=${4:-20}
PROJ=/home/teddyminelab/openarm_isaaclab
case "$FRUIT" in banana) BASKET=yellow;; *) BASKET=red;; esac
CKPT_PATH=$PROJ/outputs/finetune_act_$DATASET/checkpoints/$CKPT/pretrained_model
BASE=$PROJ/outputs/evalview_act_${DATASET}_${CKPT}
LOG_CSV=${BASE}.csv          # EEF frame log（scene 靠 EEF_LOG env 寫）
MP4=${BASE}_3cam.mp4         # 三相機並排
cd "$PROJ" || exit 1
[ -d "$CKPT_PATH" ] || { echo "ckpt 不存在: $CKPT_PATH"; exit 1; }

pkill -9 -f 'act_bridge_node' 2>/dev/null
pkill -9 -f 'run_handover_eval_scen' 2>/dev/null
pkill -9 -f 'isaaclab.sh' 2>/dev/null
pkill -9 -f 'benchmark_handover' 2>/dev/null
pkill -9 -f 'record_3cam' 2>/dev/null
sleep 4
rm -f /tmp/openarm_episode.json "$LOG_CSV" "$MP4" "${MP4}.timestamps.csv"

echo "[recview-act] $FRUIT→$BASKET  $DATASET/$CKPT  N=$NEPS  → log=$LOG_CSV  mp4=$MP4"
EEF_LOG="$LOG_CSV" setsid bash scripts/isaac/run_handover_eval_scene.sh > /tmp/recview_scene.log 2>&1 &
for i in $(seq 1 180); do
  s=$(python3 -c "import json;print(json.load(open('/tmp/openarm_episode.json')).get('state',''))" 2>/dev/null)
  [ "$s" = "idle" ] && { echo "[recview-act] scene ready i=$i"; break; }
  sleep 5
done
sleep 5
source /opt/ros/jazzy/setup.bash 2>/dev/null
source "$PROJ/ros2_ws/install/setup.bash" 2>/dev/null
echo "[recview-act] launch ACT bridge"
setsid bash policy_bridges/run_act_bridge.sh --ckpt "$CKPT_PATH" \
   --save_snapshot > /tmp/recview_bridge.log 2>&1 &
for i in $(seq 1 72); do
  grep -qE "ACT bridge ready|ACT ready" /tmp/recview_bridge.log 2>/dev/null && { echo "[recview-act] bridge ready i=$i"; break; }
  sleep 5
done
sleep 8
echo "[recview-act] start 3cam recorder → $MP4"
python3 scripts/tools/record_3cam.py --out "$MP4" --fps 30 > /tmp/recview_rec.log 2>&1 &
REC_PID=$!
sleep 2
echo "[recview-act] benchmark N=$NEPS"
python3 scripts/tools/benchmark_handover.py --fruit "$FRUIT" --basket "$BASKET" --n_episodes "$NEPS" --tag "recview_act_${CKPT}" 2>&1 | tee /tmp/recview_orch.log
echo "[recview-act] stop recorder (flush mp4)"
kill -TERM $REC_PID 2>/dev/null
for i in $(seq 1 10); do kill -0 $REC_PID 2>/dev/null || break; sleep 1; done
kill -9 $REC_PID 2>/dev/null
pkill -9 -f 'act_bridge_node' 2>/dev/null
pkill -9 -f 'run_handover_eval_scen' 2>/dev/null
pkill -9 -f 'isaaclab.sh' 2>/dev/null
echo "[recview-act] ===== DONE ====="
grep -aE '\[DONE\]' /tmp/recview_orch.log
echo "[recview-act] log=$LOG_CSV  mp4=$MP4 ($(grep -a wrote /tmp/recview_rec.log | tail -1))"
