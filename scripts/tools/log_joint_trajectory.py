#!/usr/bin/env python3
"""
Joint trajectory logger — 同時錄 robot state（實測）跟 bridge command（推論輸出）到 CSV。

訂閱：
  /Isaac/isaac_joint_states  → Isaac 發的實測 joint position (state)
  /Isaac/isaac_joint_command → bridge 推論寫的目標 joint position (action)
  /tmp/openarm_episode.json  → orchestrator 寫的 episode boundary（輪詢，no ROS）

輸出：
  outputs/joint_logs/<timestamp>_<policy>.csv
  欄位：
    t_sec       ROS 時間（從第一個 message 起算）
    source      "state" 或 "command"
    ep_idx      當前 episode（0-based；未開始為 -1）
    ep_state    "ready" / "running" / "done" / "reset"
    cube_x cube_y cube_z   cube XY（從 episode.json 帶）
    j_<name>... 每個 joint 一欄

用法（不影響現有 3 個 terminal）：
  python3 scripts/tools/log_joint_trajectory.py --policy pi05_lora_v1 --duration 600
"""
import argparse
import csv
import json
import time
from pathlib import Path
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

EPISODE_FILE = Path("/tmp/openarm_episode.json")
DEFAULT_LOG_DIR = Path("/home/teddyminelab/openarm_isaaclab/outputs/joint_logs")


class JointLogger(Node):
    def __init__(self, csv_path: Path):
        super().__init__("joint_logger")
        self.csv_path = csv_path

        self.csv_file = open(csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.header_written = False
        self.joint_names = None
        self.t0 = None

        # Episode metadata (polled from /tmp/openarm_episode.json)
        self.ep_idx = -1
        self.ep_state = "unknown"
        self.cube_xyz = [0.0, 0.0, 0.0]

        self.sub_state = self.create_subscription(
            JointState, "/Isaac/isaac_joint_states",
            lambda msg: self._on_joint(msg, "state"), 30)
        self.sub_cmd = self.create_subscription(
            JointState, "/Isaac/isaac_joint_command",
            lambda msg: self._on_joint(msg, "command"), 30)

        # 5 Hz 輪詢 episode file
        self.create_timer(0.2, self._poll_episode)

        self.n_state = 0
        self.n_cmd = 0
        self.create_timer(2.0, self._stats)

        self.get_logger().info(f"Logging to {csv_path}")

    def _poll_episode(self):
        if not EPISODE_FILE.exists():
            return
        try:
            ctrl = json.loads(EPISODE_FILE.read_text())
            self.ep_state = ctrl.get("state", self.ep_state)
            if "episode" in ctrl:
                self.ep_idx = int(ctrl["episode"])
            if "cube" in ctrl and len(ctrl["cube"]) >= 3:
                self.cube_xyz = list(ctrl["cube"][:3])
        except (json.JSONDecodeError, OSError):
            pass

    def _on_joint(self, msg: JointState, source: str):
        if self.t0 is None:
            self.t0 = time.time()

        if not self.header_written and msg.name:
            self.joint_names = list(msg.name)
            header = ["t_sec", "source", "ep_idx", "ep_state",
                      "cube_x", "cube_y", "cube_z"] + \
                     [f"j_{n}" for n in self.joint_names]
            self.writer.writerow(header)
            self.header_written = True
            self.get_logger().info(f"Joint names ({len(self.joint_names)}): {self.joint_names}")

        # 如果 message 跟 header joint 數量不對齊，pad 或 truncate
        pos = list(msg.position)
        if self.joint_names and len(pos) < len(self.joint_names):
            pos += [None] * (len(self.joint_names) - len(pos))
        elif self.joint_names and len(pos) > len(self.joint_names):
            pos = pos[:len(self.joint_names)]

        row = [f"{time.time() - self.t0:.4f}", source, self.ep_idx, self.ep_state,
               f"{self.cube_xyz[0]:.4f}", f"{self.cube_xyz[1]:.4f}", f"{self.cube_xyz[2]:.4f}"] + \
              [f"{p:.6f}" if p is not None else "" for p in pos]
        self.writer.writerow(row)

        if source == "state":
            self.n_state += 1
        else:
            self.n_cmd += 1

    def _stats(self):
        self.csv_file.flush()
        self.get_logger().info(
            f"Recorded: state={self.n_state}  command={self.n_cmd}  "
            f"ep={self.ep_idx}/{self.ep_state}")

    def close(self):
        self.csv_file.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="unknown",
                    help="標記 policy 名稱（會放進檔名）")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="自動停止秒數，0=直到 Ctrl+C")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_LOG_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.out_dir / f"{ts}_{args.policy}.csv"

    rclpy.init()
    node = JointLogger(csv_path)
    try:
        if args.duration > 0:
            end = time.time() + args.duration
            while rclpy.ok() and time.time() < end:
                rclpy.spin_once(node, timeout_sec=0.1)
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, closing.")
    finally:
        node.close()
        node.destroy_node()
        # 先印 path（SIGINT 可能已 shutdown context，等等 rclpy.shutdown() raise 蓋掉 print）
        print(f"\n✅ Wrote {csv_path}", flush=True)
        try:
            rclpy.shutdown()
        except Exception:
            pass  # context already shut down by SIGINT handler


if __name__ == "__main__":
    main()
