#!/usr/bin/env python3
"""
log_gripper_diag.py — 夾爪診斷專用 logger（非侵入，當第 4 個 terminal 跑）。

專注左右兩臂的 4 個 finger joint，同時錄 position / velocity / effort，
用來客觀量化「mimic 指被物體推回去」與「maxForce 10→333 修復」。

訂閱：
  /Isaac/isaac_joint_states   → Isaac 發的實測 state（含 mimic finger_joint2）
  /Isaac/isaac_joint_command  → MoveIt/bridge 寫的目標 position
輪詢：
  /tmp/openarm_episode.json   → episode 邊界 + cube 座標（與既有 logger 一致）

輸出 CSV 欄位（L1=left_joint1 主動指, L2=left_joint2 mimic 指, R1/R2 右臂同理）：
  t_sec, source, ep_idx, ep_state, cube_x, cube_y, cube_z,
  L1_pos,L1_vel,L1_eff, L2_pos,L2_vel,L2_eff, R1_pos,R1_vel,R1_eff, R2_pos,R2_vel,R2_eff

用法（先 source ros2_ws/install/setup.bash）：
  python3 scripts/tools/log_gripper_diag.py --tag maxforce_fixed --duration 300
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
DEFAULT_LOG_DIR = Path("/home/teddyminelab/openarm_isaaclab/outputs/gripper_diag")

# CSV 欄位順序 → (joint 名稱)
JOINTS = {
    "L1": "openarm_left_finger_joint1",
    "L2": "openarm_left_finger_joint2",
    "R1": "openarm_right_finger_joint1",
    "R2": "openarm_right_finger_joint2",
}


class GripperDiagLogger(Node):
    def __init__(self, csv_path: Path):
        super().__init__("gripper_diag_logger")
        self.csv_file = open(csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        header = ["t_sec", "source", "ep_idx", "ep_state", "cube_x", "cube_y", "cube_z"]
        for k in JOINTS:
            header += [f"{k}_pos", f"{k}_vel", f"{k}_eff"]
        self.writer.writerow(header)

        self.t0 = None
        self.ep_idx = -1
        self.ep_state = "unknown"
        self.cube_xyz = [0.0, 0.0, 0.0]
        self.n_state = 0
        self.n_cmd = 0
        self.eff_seen = False

        self.create_subscription(
            JointState, "/Isaac/isaac_joint_states",
            lambda m: self._on_joint(m, "state"), 30)
        self.create_subscription(
            JointState, "/Isaac/isaac_joint_command",
            lambda m: self._on_joint(m, "command"), 30)
        self.create_timer(0.2, self._poll_episode)
        self.create_timer(2.0, self._stats)
        self.get_logger().info(f"Logging gripper diag to {csv_path}")

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
        idx = {n: i for i, n in enumerate(msg.name)}

        def get(arr, i):
            return arr[i] if (i is not None and i < len(arr)) else None

        row = [f"{time.time() - self.t0:.4f}", source, self.ep_idx, self.ep_state,
               f"{self.cube_xyz[0]:.4f}", f"{self.cube_xyz[1]:.4f}", f"{self.cube_xyz[2]:.4f}"]
        for jname in JOINTS.values():
            i = idx.get(jname)
            p = get(msg.position, i)
            v = get(msg.velocity, i)
            e = get(msg.effort, i)
            if e is not None and e != 0.0:
                self.eff_seen = True
            row += [f"{p:.6f}" if p is not None else "",
                    f"{v:.6f}" if v is not None else "",
                    f"{e:.6f}" if e is not None else ""]
        self.writer.writerow(row)

        if source == "state":
            self.n_state += 1
        else:
            self.n_cmd += 1

    def _stats(self):
        self.csv_file.flush()
        self.get_logger().info(
            f"state={self.n_state} command={self.n_cmd} "
            f"ep={self.ep_idx}/{self.ep_state} effort_published={self.eff_seen}")

    def close(self):
        self.csv_file.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run", help="標記（放進檔名，例如 maxforce_fixed / before_fix）")
    ap.add_argument("--duration", type=float, default=0.0, help="自動停止秒數，0=Ctrl+C")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_LOG_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.out_dir / f"{ts}_{args.tag}.csv"

    rclpy.init()
    node = GripperDiagLogger(csv_path)
    try:
        if args.duration > 0:
            end = time.time() + args.duration
            while rclpy.ok() and time.time() < end:
                rclpy.spin_once(node, timeout_sec=0.1)
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        print(f"\n✅ Wrote {csv_path}", flush=True)
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
