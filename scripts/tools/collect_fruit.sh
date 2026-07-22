#!/bin/bash
# 通用 handover 收集（apple→red / strawberry→red / banana→yellow）。
# 用法: bash scripts/tools/collect_fruit.sh <fruit> <basket> <num_success> [output_dir] [jitter]
#   output_dir 預設 datasets_handover；第 5 參給 "jitter" 開放籃隨機偏移（收 return-recovery 多樣性資料）
FRUIT=${1:-apple}
BASKET=${2:-red}
NSUCC=${3:-100}
OUTDIR=${4:-datasets_handover}
JITTER=${5:-}
JITTER_ARG=""
[ "$JITTER" = "jitter" ] && JITTER_ARG="--place_jitter"
PROJ=/home/teddyminelab/openarm_isaaclab
cd "$PROJ" || exit 1
pkill -9 -f collect_handover_demos 2>/dev/null
pkill -9 -f run_handover_episodes 2>/dev/null
pkill -9 -f "moveit_ros_move_group/move_group" 2>/dev/null
pkill -9 -f ros2_control_node 2>/dev/null
sleep 4
rm -f /tmp/openarm_episode.json "/tmp/collect_${FRUIT}_t1.log" "/tmp/collect_${FRUIT}_t2.log"
BEFORE=$(ls -d "$OUTDIR"/*/ 2>/dev/null | wc -l)
echo "[orch] BEFORE=$BEFORE; launch Isaac collect $FRUIT→$BASKET num_success=$NSUCC outdir=$OUTDIR jitter='${JITTER_ARG}'"
conda run -n env_isaaclab "$HOME/IsaacLab-2.3.0/isaaclab.sh" -p scripts/isaac/collect_handover_demos.py \
    --enable_cameras --fruit "$FRUIT" --basket "$BASKET" --num_success "$NSUCC" --output_dir "$OUTDIR" > "/tmp/collect_${FRUIT}_t1.log" 2>&1 &
T1=$!
echo "[orch] T1=$T1 waiting Isaac ready..."
ready=0
for i in $(seq 1 150); do
  NOW=$(ls -d "$OUTDIR"/*/ 2>/dev/null | wc -l)
  [ "$NOW" -gt "$BEFORE" ] && { ready=1; echo "[orch] Isaac ready i=$i"; break; }
  kill -0 $T1 2>/dev/null || { echo "[orch] T1 died early"; tail -n 25 "/tmp/collect_${FRUIT}_t1.log"; exit 1; }
  sleep 5
done
[ $ready -eq 0 ] && { echo "[orch] Isaac not ready in 12.5min"; tail -n 25 "/tmp/collect_${FRUIT}_t1.log"; kill -9 $T1 2>/dev/null; exit 1; }
sleep 3
echo "[orch] launch run_handover_episodes $FRUIT"
source /opt/ros/jazzy/setup.bash 2>/dev/null
source "$PROJ/ros2_ws/install/setup.bash" 2>/dev/null
python3 scripts/ros2/run_handover_episodes.py --fruit "$FRUIT" --basket "$BASKET" $JITTER_ARG > "/tmp/collect_${FRUIT}_t2.log" 2>&1 &
T2=$!
echo "[orch] T2=$T2 collecting until all_done..."
for i in $(seq 1 3600); do
  grep -q "all_done" /tmp/openarm_episode.json 2>/dev/null && { echo "[orch] all_done!"; sleep 3; break; }
  kill -0 $T1 2>/dev/null || { echo "[orch] T1 exited"; break; }
  sleep 5
done
pkill -9 -f collect_handover_demos 2>/dev/null; pkill -9 -f isaaclab.sh 2>/dev/null
kill $T2 2>/dev/null; pkill -9 -f run_handover_episodes 2>/dev/null
pkill -9 -f "moveit_ros_move_group/move_group" 2>/dev/null; pkill -9 -f ros2_control_node 2>/dev/null
echo "[orch] COLLECT DONE"
