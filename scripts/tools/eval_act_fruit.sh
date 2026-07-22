#!/bin/bash
# 通用單任務 ACT 閉環評測（改自 eval_fruit.sh 的 SmolVLA 版）。
# 用法: bash scripts/tools/eval_act_fruit.sh <fruit> <dataset> <ckpt6> <n_eps> [home_wait_s]
# 例: bash scripts/tools/eval_act_fruit.sh apple handover_apple 040000 10
# ★ 與 SmolVLA 版差異：ckpt 指向 finetune_act_*、換 run_act_bridge、
#   ACT 無 --task/--execution_horizon/--disable_gripper_snap（無語言/無 RTC/無 gripper snap），
#   bridge ready marker = "ACT bridge ready"。場景與 benchmark 與 policy 無關，沿用同一套。
FRUIT=${1:?用法: eval_act_fruit.sh <fruit> <dataset> <ckpt> <n> [home_wait]}
DATASET=${2:?缺 dataset}
CKPT=${3:-040000}
NEPS=${4:-10}
HW=${5:-8}   # 入籃後等雙臂回 home 的秒數
PROJ=/home/teddyminelab/openarm_isaaclab
# basket: banana→yellow, 其他→red（跟 collect TASK_DESC 一致；ACT 不吃 task，這裡只給 benchmark 判 success）
case "$FRUIT" in banana) BASKET=yellow;; *) BASKET=red;; esac
CKPT_PATH=$PROJ/outputs/finetune_act_$DATASET/checkpoints/$CKPT/pretrained_model
cd "$PROJ" || exit 1
[ -d "$CKPT_PATH" ] || { echo "ckpt 不存在: $CKPT_PATH"; exit 1; }

pkill -9 -f 'act_bridge_node' 2>/dev/null
pkill -9 -f 'run_handover_eval_scen' 2>/dev/null
pkill -9 -f 'isaaclab.sh' 2>/dev/null
pkill -9 -f 'benchmark_handover' 2>/dev/null
sleep 4
rm -f /tmp/openarm_episode.json /tmp/eval_scene.log /tmp/eval_bridge_${FRUIT}.log /tmp/eval_orch_${FRUIT}.log

echo "[eval-act] $FRUIT→$BASKET  dataset=$DATASET  ckpt=$CKPT  n=$NEPS  ckpt_path=$CKPT_PATH"
setsid bash scripts/isaac/run_handover_eval_scene.sh > /tmp/eval_scene.log 2>&1 &
for i in $(seq 1 180); do
  s=$(python3 -c "import json;print(json.load(open('/tmp/openarm_episode.json')).get('state',''))" 2>/dev/null)
  [ "$s" = "idle" ] && { echo "[eval-act] scene ready i=$i"; break; }
  sleep 5
done
sleep 5
source /opt/ros/jazzy/setup.bash 2>/dev/null
source "$PROJ/ros2_ws/install/setup.bash" 2>/dev/null
echo "[eval-act] launch ACT bridge"
setsid bash policy_bridges/run_act_bridge.sh --ckpt "$CKPT_PATH" \
   --save_snapshot > /tmp/eval_bridge_${FRUIT}.log 2>&1 &
for i in $(seq 1 72); do
  grep -qE "ACT bridge ready|ACT ready" /tmp/eval_bridge_${FRUIT}.log 2>/dev/null && { echo "[eval-act] bridge ready i=$i"; break; }
  sleep 5
done
grep -aE "ACT bridge ready|ACT ready" /tmp/eval_bridge_${FRUIT}.log | tail -1 | sed -E 's/.*bridge\]: //'
sleep 10
echo "[eval-act] benchmark $FRUIT n=$NEPS"
python3 scripts/tools/benchmark_handover.py --fruit "$FRUIT" --basket "$BASKET" --n_episodes "$NEPS" --home_wait_s "$HW" --tag "act_${DATASET}_${CKPT}_hw${HW}" 2>&1 | tee /tmp/eval_orch_${FRUIT}.log
echo "[eval-act] DONE — 殺 scene/bridge"
pkill -9 -f 'act_bridge_node' 2>/dev/null
pkill -9 -f 'run_handover_eval_scen' 2>/dev/null
pkill -9 -f 'isaaclab.sh' 2>/dev/null
echo "[eval-act] ALL DONE"
