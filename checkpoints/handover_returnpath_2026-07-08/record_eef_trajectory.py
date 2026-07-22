#!/usr/bin/env python3
"""record_eef_trajectory.py ─ 記錄雙臂 EEF (hand_tcp) 世界座標 + 姿態軌跡到 CSV。

調 handover 軌跡時用來『把路徑記下來』，之後用 plot_eef_trajectory.py 畫成 3D 圖目視。
訂閱 TF world→openarm_{left,right}_hand_tcp，固定頻率取樣。

用法（第四終端，T1/T2 在跑、handover 開始前先啟動）：
  source ros2_ws/install/setup.bash
  python3 scripts/tools/record_eef_trajectory.py --out outputs/eef_traj.csv
  # 然後在 T3 跑 bimanual_handover.py；跑完回這裡 Ctrl-C（或 kill）→ 自動存 CSV。
"""
import argparse
import csv
import math
import signal
import time

import rclpy
import rclpy.time
import rclpy.duration
from rclpy.node import Node
import tf2_ros
from tf2_ros import TransformException

FRAME = "world"
TCPS = {"left": "openarm_left_hand_tcp", "right": "openarm_right_hand_tcp"}


def quat_to_rpy(x, y, z, w):
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class Recorder(Node):
    def __init__(self, args):
        super().__init__("eef_recorder")
        self.args = args
        self.buf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buf, self)
        self.rows = []
        self.t0 = None
        self.create_timer(1.0 / args.rate, self.tick)
        self.get_logger().info(
            f"recording @ {args.rate}Hz -> {args.out}  (Ctrl-C / kill to stop & save)")

    def tick(self):
        now = time.time()
        if self.t0 is None:
            self.t0 = now
        row = {"t": round(now - self.t0, 4)}
        got = False
        for side, tcp in TCPS.items():
            try:
                tr = self.buf.lookup_transform(
                    FRAME, tcp, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.05))
                p, q = tr.transform.translation, tr.transform.rotation
                r, pi, ya = quat_to_rpy(q.x, q.y, q.z, q.w)
                row[f"{side}_x"], row[f"{side}_y"], row[f"{side}_z"] = (
                    round(p.x, 5), round(p.y, 5), round(p.z, 5))
                row[f"{side}_roll"], row[f"{side}_pitch"], row[f"{side}_yaw"] = (
                    round(r, 4), round(pi, 4), round(ya, 4))
                got = True
            except TransformException:
                pass
        if got:
            self.rows.append(row)

    def save(self):
        if not self.rows:
            self.get_logger().warn("no rows recorded (TF 沒抓到？確認 T1/T2 在跑)")
            return
        cols = ["t"]
        for side in TCPS:
            cols += [f"{side}_{k}" for k in ("x", "y", "z", "roll", "pitch", "yaw")]
        with open(self.args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, restval="")
            w.writeheader()
            w.writerows(self.rows)
        self.get_logger().info(f"saved {len(self.rows)} rows -> {self.args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/eef_traj.csv")
    ap.add_argument("--rate", type=float, default=50.0)
    a = ap.parse_args()

    rclpy.init()
    node = Recorder(a)

    # 讓背景 kill (SIGTERM) 也能觸發存檔（不只 Ctrl-C）
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
