#!/bin/bash
# 通用單任務閉環評測。用法: bash scripts/tools/eval_fruit.sh <fruit> <dataset> <ckpt6> <n_eps> <exec_horizon>
# 例: bash scripts/tools/eval_fruit.sh strawberry strawberry_v1 040000 10 25
FRUIT=${1:?用法: eval_fruit.sh <fruit> <dataset> <ckpt> <n> <eh>}
DATASET=${2:?缺 dataset}
CKPT=${3:-040000}
NEPS=${4:-10}
EH=${5:-25}
HW=${6:-8}   # 入籃後等雙臂回 home 的秒數（default 8；衡量回位要拉大如 25）
PROJ=/home/teddyminelab/openarm_isaaclab
# basket: banana→yellow, 其他→red（跟 collect TASK_DESC 一致）
case "$FRUIT" in banana) BASKET=yellow;; *) BASKET=red;; esac
TASK="pick up the $FRUIT and place it into the $BASKET basket"
CKPT_PATH=$PROJ/outputs/finetune_smolvla_$DATASET/checkpoints/$CKPT/pretrained_model
cd "$PROJ" || exit 1
[ -d "$CKPT_PATH" ] || { echo "ckpt 不存在: $CKPT_PATH"; exit 1; }

pkill -9 -f 'smolvla_rtc_bridge_node' 2>/dev/null
pkill -9 -f 'run_handover_eval_scen' 2>/dev/null
pkill -9 -f 'isaaclab.sh' 2>/dev/null
pkill -9 -f 'benchmark_handover' 2>/dev/null
sleep 4
rm -f /tmp/openarm_episode.json /tmp/eval_scene.log /tmp/eval_bridge_${FRUIT}.log /tmp/eval_orch_${FRUIT}.log

echo "[eval] $FRUIT→$BASKET  dataset=$DATASET  ckpt=$CKPT  n=$NEPS  EH=$EH"
echo "[eval] task=\"$TASK\""
setsid bash scripts/isaac/run_handover_eval_scene.sh > /tmp/eval_scene.log 2>&1 &
for i in $(seq 1 180); do
  s=$(python3 -c "import json;print(json.load(open('/tmp/openarm_episode.json')).get('state',''))" 2>/dev/null)
  [ "$s" = "idle" ] && { echo "[eval] scene ready i=$i"; break; }
  sleep 5
done
sleep 5
source /opt/ros/jazzy/setup.bash 2>/dev/null
source "$PROJ/ros2_ws/install/setup.bash" 2>/dev/null
echo "[eval] launch bridge"
setsid bash policy_bridges/run_smolvla_rtc_bridge.sh --ckpt "$CKPT_PATH" \
   --task "$TASK" --disable_gripper_snap \
   --execution_horizon "$EH" --save_snapshot > /tmp/eval_bridge_${FRUIT}.log 2>&1 &
for i in $(seq 1 72); do
  grep -qE "RTC enabled|Waiting for first RTC|inference thread" /tmp/eval_bridge_${FRUIT}.log 2>/dev/null && { echo "[eval] bridge ready i=$i"; break; }
  sleep 5
done
grep -a "RTC enabled" /tmp/eval_bridge_${FRUIT}.log | tail -1 | sed -E 's/.*bridge\]: //'
sleep 10
echo "[eval] benchmark $FRUIT n=$NEPS"
python3 scripts/tools/benchmark_handover.py --fruit "$FRUIT" --basket "$BASKET" --n_episodes "$NEPS" --home_wait_s "$HW" --tag "${DATASET}_${CKPT}_eh${EH}_hw${HW}" 2>&1 | tee /tmp/eval_orch_${FRUIT}.log
echo "[eval] DONE — 殺 scene/bridge"
pkill -9 -f 'smolvla_rtc_bridge_node' 2>/dev/null
pkill -9 -f 'run_handover_eval_scen' 2>/dev/null
pkill -9 -f 'isaaclab.sh' 2>/dev/null
echo "[eval] ALL DONE"
