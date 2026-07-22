#!/bin/bash
# 編排：清理 → 起 Isaac → 等「新 results 目錄出現」(就緒訊號,不依賴緩衝log) → 起 watcher → 等跑完
NEPS=${1:-2}
FRUIT=${2:-apple}
BASKET=${3:-red}
PROJ=/home/teddyminelab/openarm_isaaclab
cd "$PROJ" || exit 1

pkill -9 -f run_handover_benchmark 2>/dev/null
pkill -9 -f run_handover_episodes 2>/dev/null
pkill -9 -f "openarm_bimanual_moveit_config demo.launch" 2>/dev/null
pkill -9 -f "moveit_ros_move_group/move_group" 2>/dev/null
pkill -9 -f ros2_control_node 2>/dev/null
pkill -9 -f isaac_joint_command_merger 2>/dev/null
sleep 4
rm -f /tmp/openarm_episode.json

BEFORE=$(ls -d results_handover_bench/*/ 2>/dev/null | wc -l)

echo "[orch] launch Isaac ($NEPS eps)..."
conda run -n env_isaaclab "$HOME/IsaacLab-2.3.0/isaaclab.sh" -p scripts/isaac/run_handover_benchmark.py \
    --num_episodes "$NEPS" --fruit "$FRUIT" --basket "$BASKET" > /tmp/bench_t1.log 2>&1 &
T1=$!

echo "[orch] waiting for new results dir (Isaac ready)..."
ready=0
for i in $(seq 1 96); do   # up to 8 min
  NOW=$(ls -d results_handover_bench/*/ 2>/dev/null | wc -l)
  [ "$NOW" -gt "$BEFORE" ] && { ready=1; echo "[orch] Isaac ready"; break; }
  kill -0 $T1 2>/dev/null || { echo "[orch] T1 died early"; exit 1; }
  sleep 5
done
[ $ready -eq 0 ] && { echo "[orch] Isaac not ready in time"; kill -9 $T1; exit 1; }

sleep 3
echo "[orch] launch watcher..."
source "$PROJ/ros2_ws/install/setup.bash"
echo "[orch] fruit=$FRUIT"
python3 scripts/ros2/run_handover_episodes.py --fruit "$FRUIT" --basket "$BASKET" > /tmp/bench_t2.log 2>&1 &
T2=$!

echo "[orch] running... (waiting for json all_done, up to 60 min)"
for i in $(seq 1 720); do
  if grep -q "all_done" /tmp/openarm_episode.json 2>/dev/null; then echo "[orch] all_done detected"; sleep 3; break; fi
  kill -0 $T1 2>/dev/null || { echo "[orch] T1 exited"; break; }
  sleep 5
done

# 強制收掉 Isaac（其 close() 可能 hang）+ watcher
pkill -9 -f run_handover_benchmark 2>/dev/null
pkill -9 -f isaaclab.sh 2>/dev/null
kill $T2 2>/dev/null
pkill -9 -f run_handover_episodes 2>/dev/null
pkill -9 -f "openarm_bimanual_moveit_config demo.launch" 2>/dev/null
pkill -9 -f "moveit_ros_move_group/move_group" 2>/dev/null
pkill -9 -f ros2_control_node 2>/dev/null
pkill -9 -f isaac_joint_command_merger 2>/dev/null
echo "[orch] BENCH DONE"
