#!/usr/bin/env python3
"""
ROS2 端的 episode controller，搭配 Isaac Lab collect_demonstrations.py 使用。

功能：
  監控 /tmp/openarm_episode.json 的狀態，每當 Isaac Lab 寫入
  {"state": "ready", "cube": [x, y, z]} 時，呼叫 platform_grasp
  執行 MoveIt2 夾取，完成後寫回 {"state": "done"}。

使用方式（需要先啟動 ros2_control + MoveIt2）：
  Terminal 2：
    source ~/openarm_isaaclab/ros2_ws/install/setup.bash
    ros2 launch openarm_bringup openarm.bimanual.launch.py \
        use_isaac_sim:=true robot_controller:=joint_trajectory_controller

  Terminal 3：
    source ~/openarm_isaaclab/ros2_ws/install/setup.bash
    ros2 launch openarm_bimanual_moveit_config demo.launch.py

  Terminal 4（本腳本）：
    source ~/openarm_isaaclab/ros2_ws/install/setup.bash
    python3 scripts/ros2_episode_controller.py

注意：本腳本需要在已 source ROS2 workspace 的環境中執行（Python 3.12）。
"""

import json
import os
import subprocess
import sys
import time

# ── 設定 ──────────────────────────────────────────────────────────────────────

EPISODE_FILE = "/tmp/openarm_episode.json"

# platform_grasp.py 的路徑（本專案內，不依賴 ros2_ws binary）
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PLATFORM_GRASP_SCRIPT = os.path.join(SCRIPTS_DIR, "platform_grasp.py")

# 放置目標：籃口上方釋放，讓方塊自由落體落入籃子
# PLACE_Z = 0.30：比籃子頂部（0.246）高 5.4cm，方塊落體入籃，不需要伸進去
PLACE_X = 0.298
PLACE_Y = 0.020
PLACE_Z = 0.30

# 每次夾取的最長等待時間（秒）
GRASP_TIMEOUT_S = 60.0

# 等待下一個 ready 的輪詢間隔（秒）
POLL_INTERVAL_S = 0.3

# ── 工具函式 ─────────────────────────────────────────────────────────────────

def read_episode() -> dict:
    try:
        with open(EPISODE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"state": "idle"}


def write_episode(data: dict):
    with open(EPISODE_FILE, "w") as f:
        json.dump(data, f)


LOG_DIR = "/tmp/grasp_logs"
os.makedirs(LOG_DIR, exist_ok=True)


def run_platform_grasp(cube_x: float, cube_y: float, cube_z: float, ep_idx: int) -> bool:
    """
    呼叫 platform_grasp.py 執行夾取。
    stdout/stderr 同時輸出到終端及 /tmp/grasp_logs/ep_NNN.log。
    """
    cmd = [
        sys.executable, PLATFORM_GRASP_SCRIPT,
        f"--cube_x={cube_x:.5f}",
        f"--cube_y={cube_y:.5f}",
        f"--cube_z={cube_z:.5f}",
        f"--place_x={PLACE_X:.5f}",
        f"--place_y={PLACE_Y:.5f}",
        f"--place_z={PLACE_Z:.5f}",
    ]

    log_path = os.path.join(LOG_DIR, f"ep_{ep_idx:03d}.log")
    print(f"[ROS2] Running: {' '.join(cmd)}")
    print(f"[ROS2] Log: {log_path}")

    try:
        with open(log_path, "w") as lf:
            lf.write(f"CMD: {' '.join(cmd)}\n\n")
            lf.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                lf.write(line)
            proc.wait(timeout=GRASP_TIMEOUT_S)
        print(f"[ROS2] platform_grasp exited with code {proc.returncode}")
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"[ROS2] platform_grasp timed out after {GRASP_TIMEOUT_S}s")
        return False
    except FileNotFoundError:
        print("[ROS2] ERROR: 'ros2' command not found. Did you source the ROS2 workspace?")
        sys.exit(1)


# ── 主迴圈 ────────────────────────────────────────────────────────────────────

def main():
    print("[ROS2] Episode controller started.")
    print(f"[ROS2] Watching: {EPISODE_FILE}")
    print("[ROS2] Waiting for Isaac Lab to write 'ready'...\n")

    # 初始化（如果檔案不存在就建立）
    if not os.path.exists(EPISODE_FILE):
        write_episode({"state": "idle"})

    ep_count = 0

    while True:
        ctrl = read_episode()
        state = ctrl.get("state", "idle")

        if state == "ready":
            cube = ctrl.get("cube", [0.25, 0.15, 0.213])
            ep_idx = ctrl.get("episode", ep_count)
            cx, cy, cz = float(cube[0]), float(cube[1]), float(cube[2])

            print(f"[ROS2] ─── Episode {ep_idx:03d} ───────────────────────────────")
            print(f"[ROS2] Cube at ({cx:.3f}, {cy:.3f}, {cz:.3f})")

            # 標記為執行中
            write_episode({"state": "running", "episode": ep_idx})

            # 執行 MoveIt2 夾取
            run_platform_grasp(cx, cy, cz, ep_idx)

            # platform_grasp 用 os._exit(0) 強制結束，move_group 需要幾秒
            # 讓 action server 清理未處理的請求，避免下一個 episode 被 reject。
            time.sleep(10.0)

            # 完成：Isaac Lab 側會檢查方塊是否真的被抬起
            write_episode({"state": "done", "episode": ep_idx})
            print(f"[ROS2] Episode {ep_idx:03d} done.\n")

            ep_count += 1

        elif state in ("done", "idle"):
            # 等待下一個 episode
            pass
        else:
            # running 狀態：不應發生（兩次收到 ready 前 running 應已清除）
            pass

        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
