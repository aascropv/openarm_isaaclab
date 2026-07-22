#!/bin/bash
# 跑「一集」閉環 eval 並錄俯視鏡頭成 mp4（給人看行為用）。改自 eval_fruit.sh，N=1。
# 用法: bash scripts/tools/eval_record_one.sh <fruit> <dataset> <ckpt6> <exec_horizon>
# 例: bash scripts/tools/eval_record_one.sh banana handover_multitask_v1 040000 25
FRUIT=${1:?用法: eval_record_one.sh <fruit> <dataset> <ckpt> <eh>}
DATASET=${2:?缺 dataset}
CKPT=${3:-040000}
EH=${4:-25}
PROJ=/home/teddyminelab/openarm_isaaclab
case "$FRUIT" in banana) BASKET=yellow;; *) BASKET=red;; esac
TASK="pick up the $FRUIT and place it into the $BASKET basket"
CKPT_PATH=$PROJ/outputs/finetune_smolvla_$DATASET/checkpoints/$CKPT/pretrained_model
MP4=$PROJ/outputs/eval_record_${FRUIT}_${DATASET}_${CKPT}.mp4
cd "$PROJ" || exit 1
[ -d "$CKPT_PATH" ] || { echo "ckpt 不存在: $CKPT_PATH"; exit 1; }

pkill -9 -f 'smolvla_rtc_bridge_node' 2>/dev/null
pkill -9 -f 'run_handover_eval_scen' 2>/dev/null
pkill -9 -f 'isaaclab.sh' 2>/dev/null
pkill -9 -f 'benchmark_handover' 2>/dev/null
pkill -9 -f 'record_camera' 2>/dev/null
sleep 4
rm -f /tmp/openarm_episode.json "$MP4"

echo "[rec] $FRUIT→$BASKET  dataset=$DATASET  ckpt=$CKPT  N=1  → $MP4"
setsid bash scripts/isaac/run_handover_eval_scene.sh > /tmp/rec_scene.log 2>&1 &
for i in $(seq 1 180); do
  s=$(python3 -c "import json;print(json.load(open('/tmp/openarm_episode.json')).get('state',''))" 2>/dev/null)
  [ "$s" = "idle" ] && { echo "[rec] scene ready i=$i"; break; }
  sleep 5
done
sleep 5
source /opt/ros/jazzy/setup.bash 2>/dev/null
source "$PROJ/ros2_ws/install/setup.bash" 2>/dev/null
echo "[rec] launch bridge"
setsid bash policy_bridges/run_smolvla_rtc_bridge.sh --ckpt "$CKPT_PATH" \
   --task "$TASK" --disable_gripper_snap \
   --execution_horizon "$EH" > /tmp/rec_bridge.log 2>&1 &
for i in $(seq 1 72); do
  grep -qE "RTC enabled|Waiting for first RTC|inference thread" /tmp/rec_bridge.log 2>/dev/null && { echo "[rec] bridge ready i=$i"; break; }
  sleep 5
done
sleep 8

echo "[rec] start recorder → $MP4"
python3 scripts/tools/record_camera.py --topic /camera/overhead/image_raw --out "$MP4" --fps 30 > /tmp/rec_recorder.log 2>&1 &
REC_PID=$!
sleep 2

echo "[rec] benchmark N=1"
python3 scripts/tools/benchmark_handover.py --fruit "$FRUIT" --basket "$BASKET" --n_episodes 1 --tag "record_${CKPT}" 2>&1 | tee /tmp/rec_orch.log

echo "[rec] stop recorder (SIGTERM, flush mp4)"
kill -TERM $REC_PID 2>/dev/null
for i in $(seq 1 10); do kill -0 $REC_PID 2>/dev/null || break; sleep 1; done
kill -9 $REC_PID 2>/dev/null

echo "[rec] teardown"
pkill -9 -f 'smolvla_rtc_bridge_node' 2>/dev/null
pkill -9 -f 'run_handover_eval_scen' 2>/dev/null
pkill -9 -f 'isaaclab.sh' 2>/dev/null
echo "[rec] ===== RESULT ====="
grep -aE '\[DONE\]|EP 000\]' /tmp/rec_orch.log
echo "[rec] mp4: $MP4  ($(ls -la "$MP4" 2>/dev/null | awk '{print $5}') bytes, $(grep -a 'wrote' /tmp/rec_recorder.log | tail -1))"