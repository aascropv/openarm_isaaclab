#!/bin/bash
# 啟動 Isaac Lab 整合所需的 ROS2 stack
# 使用方式：bash scripts/launch_ros2_stack.sh
#
# 架構（共 2 個 tmux 窗格 + Isaac Lab 另開）：
#   窗格 0：demo.launch.py use_isaac_sim:=true
#             （包含 ros2_control, move_group, isaac_joint_command_merger）
#   窗格 1：ros2_episode_controller.py
#             （監看 /tmp/openarm_episode.json，自動呼叫 platform_grasp）

set -e

WS="$HOME/openarm_isaaclab/ros2_ws"

if ! command -v tmux &>/dev/null; then
    echo "tmux 未安裝，請在兩個不同終端機分別執行以下指令："
    echo ""
    echo "  Terminal 1 (ROS2 all-in-one):"
    echo "    source $WS/install/setup.bash"
    echo "    ros2 launch openarm_bimanual_moveit_config demo.launch.py \\"
    echo "        use_isaac_sim:=true launch_rviz:=false"
    echo ""
    echo "  Terminal 2 (episode controller):"
    echo "    source $WS/install/setup.bash"
    echo "    python3 scripts/ros2/ros2_episode_controller.py"
    exit 0
fi

SESSION="openarm_ros2"

tmux kill-session -t $SESSION 2>/dev/null || true

# 窗格 0：demo.launch.py（ros2_control + move_group + merger，全部一起）
tmux new-session -d -s $SESSION -x 220 -y 50 \
    "source $WS/install/setup.bash && \
     ros2 launch openarm_bimanual_moveit_config demo.launch.py \
         use_isaac_sim:=true launch_rviz:=false; \
     read"

# 窗格 1：episode controller（等 10 秒讓 move_group 就緒）
tmux split-window -t $SESSION -v \
    "sleep 10 && source $WS/install/setup.bash && \
     python3 scripts/ros2/ros2_episode_controller.py; \
     read"

tmux select-layout -t $SESSION even-vertical

echo "tmux session '$SESSION' 已啟動。"
echo "使用 'tmux attach -t $SESSION' 查看輸出。"
echo ""
echo "ROS2 stack 就緒後（約 12 秒），在另一個終端機執行："
echo "  conda run -n env_isaaclab TERM=xterm \\"
echo "    ~/IsaacLab-2.3.0/isaaclab.sh -p scripts/collect_demonstrations.py \\"
echo "    --headless --enable_cameras --num_episodes 50"

tmux attach -t $SESSION
