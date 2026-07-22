#!/usr/bin/env python3
"""
run_handover_episodes.py ─ 雙臂手遞手 benchmark 的 ROS2 觸發端（每集重啟 move_group）

監控 /tmp/openarm_episode.json，每當 Isaac 端 run_handover_benchmark.py 寫 "ready" 就：
  1. 啟動 demo.launch.py（ros2_control + move_group + merger）→ 等 /move_action 就緒
  2. subprocess 跑一次 bimanual_handover.py（stdout 存 /tmp/handover_logs/ep_NNN.log）
  3. 殺掉整個 demo.launch 進程組 → 等 /move_action 消失
  4. 寫回 "done"

**每集重啟 move_group** 是為了徹底斷開 -26 crash 跨集級聯：實測 move_group 會在左手
step8/13 大幅轉向的硬請求機率性 CRASH，且 crash 後跨集持續（server 端，subprocess 重啟
client 救不了），導致後續整批失敗。每集給乾淨 planner 才能量到真實機制成功率。

**重要：不要再手動開 Terminal 2（demo.launch.py）——本腳本會自己管理。** 只需：
  T1: conda run -n env_isaaclab ~/IsaacLab-2.3.0/isaaclab.sh -p scripts/isaac/run_handover_benchmark.py --num_episodes 20
  T3 (本腳本): source ros2_ws/install/setup.bash; python3 scripts/ros2/run_handover_episodes.py
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

EPISODE_FILE = "/tmp/openarm_episode.json"
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
HANDOVER_SCRIPT = os.path.join(SCRIPTS_DIR, "bimanual_handover.py")
LOG_DIR = "/tmp/handover_logs"

MG_LAUNCH = ["ros2", "launch", "openarm_bimanual_moveit_config", "demo.launch.py",
             "use_isaac_sim:=true", "launch_rviz:=false"]

HANDOVER_TIMEOUT_S = 150.0
MG_READY_TIMEOUT_S = 75.0      # move_group 啟動就緒上限
MG_SETTLE_S        = 12.0      # /move_action 出現後再等控制器 spawn/active（8s 偶爾 home 規劃失敗 → 加長）
POLL_INTERVAL_S    = 0.3

os.makedirs(LOG_DIR, exist_ok=True)


def read_episode():
    try:
        with open(EPISODE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"state": "idle"}


def write_episode(data):
    with open(EPISODE_FILE, "w") as f:
        json.dump(data, f)


def move_action_up() -> bool:
    try:
        out = subprocess.run(["ros2", "action", "list"], capture_output=True, text=True, timeout=10)
        return "/move_action" in out.stdout
    except Exception:
        return False


def wait_until(cond, timeout, interval=2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


def launch_move_group(ep_idx):
    log_path = os.path.join(LOG_DIR, f"move_group_ep{ep_idx:03d}.log")
    lf = open(log_path, "w")
    # start_new_session=True → 自成 process group，方便整組 kill
    proc = subprocess.Popen(MG_LAUNCH, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"[ROS2] launched move_group (pid={proc.pid}), log: {log_path}")
    return proc


def kill_move_group(proc):
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)   # ros2 launch 收到 SIGINT 會優雅關閉子節點
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
        except Exception:
            pass
    print("[ROS2] move_group killed.")


def run_handover(basket: str, fruit: str, ep_idx: int, place_jitter: bool = False) -> bool:
    cmd = [sys.executable, HANDOVER_SCRIPT, "--basket", basket, "--fruit", fruit]
    if place_jitter:
        cmd.append("--place_jitter")
    log_path = os.path.join(LOG_DIR, f"ep_{ep_idx:03d}.log")
    print(f"[ROS2] Running: {' '.join(cmd)}  (log: {log_path})")
    try:
        with open(log_path, "w") as lf:
            lf.write(f"CMD: {' '.join(cmd)}\n\n")
            lf.flush()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                sys.stdout.write(line)
                lf.write(line)
            proc.wait(timeout=HANDOVER_TIMEOUT_S)
        print(f"[ROS2] handover exited with code {proc.returncode}")
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"[ROS2] handover timed out after {HANDOVER_TIMEOUT_S}s")
        return False


def run_episode(basket: str, fruit: str, ep_idx: int, place_jitter: bool = False):
    mg = launch_move_group(ep_idx)
    ready = wait_until(move_action_up, MG_READY_TIMEOUT_S)
    if ready:
        time.sleep(MG_SETTLE_S)   # 等 ros2_control 控制器 spawn/active
        # 標記「真正要開始跑 handover」：move_group 已就緒，接下來才是手臂運動。
        # 收集端（collect_handover_demos.py）只錄 executing→done，跳過上面 move_group 開機的 idle 幀。
        # benchmark 端只看 "done"，不受此狀態影響。
        write_episode({"state": "executing", "episode": ep_idx})
        run_handover(basket, fruit, ep_idx, place_jitter)
    else:
        print(f"[ROS2] move_group 未在 {MG_READY_TIMEOUT_S}s 內就緒，跳過 ep{ep_idx:03d}")
    kill_move_group(mg)
    # 等 /move_action 真的消失再放行下一集，確保乾淨重啟
    wait_until(lambda: not move_action_up(), timeout=20)
    time.sleep(3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--basket", choices=["red", "yellow"], default="red")
    p.add_argument("--fruit", choices=["apple", "strawberry", "banana"], default="apple")
    p.add_argument("--place_jitter", action="store_true",
                   help="放籃位置加隨機偏移（傳給 bimanual_handover）→ 收 return-recovery 多樣性資料")
    args = p.parse_args()

    print("[ROS2] Handover episode controller started (每集重啟 move_group 模式).")
    print("[ROS2] 不要手動開 demo.launch.py — 本腳本會自己管理。")
    print(f"[ROS2] Watching: {EPISODE_FILE}\n")
    if not os.path.exists(EPISODE_FILE):
        write_episode({"state": "idle"})

    while True:
        ctrl = read_episode()
        state = ctrl.get("state", "idle")

        if state == "all_done":
            print("[ROS2] all_done received, exiting.")
            break

        if state == "ready":
            ep_idx = ctrl.get("episode", 0)
            print(f"[ROS2] ─── Episode {ep_idx:03d} ───────────────────────────────")
            write_episode({"state": "running", "episode": ep_idx})
            run_episode(args.basket, args.fruit, ep_idx, args.place_jitter)
            write_episode({"state": "done", "episode": ep_idx})
            print(f"[ROS2] Episode {ep_idx:03d} done.\n")

        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
