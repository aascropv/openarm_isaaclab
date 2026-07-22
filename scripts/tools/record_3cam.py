#!/usr/bin/env python3
"""錄三個 ROS2 相機 topic → 一支並排 mp4（overhead|left|right）+ 每幀 wall 時間戳 CSV。
以 overhead 相機為 master clock：每收到一張 overhead，就把當下 left/right 最新影像並排寫一幀，
並記 (frame_idx, wall_time) 到 sidecar CSV，供 viewer 用 wall 時間跟 EEF frame log 對齊。
跑到 SIGTERM/SIGINT 為止（flush 檔）。

用法（ROS2 環境）：
  python3 scripts/tools/record_3cam.py --out /path/eval.mp4 --fps 30
"""
import argparse
import csv
import signal
import sys
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

TOPICS = {
    "overhead": "/camera/overhead/image_raw",
    "left":     "/camera/left/image_raw",
    "right":    "/camera/right/image_raw",
}


class ThreeCamRecorder(Node):
    def __init__(self, out: str, fps: float):
        super().__init__("record_3cam")
        self.bridge = CvBridge()
        self.out = out
        self.fps = fps
        self.latest = {"overhead": None, "left": None, "right": None}
        self.writer = None
        self.n = 0
        self.ts_file = open(out + ".timestamps.csv", "w", newline="")
        self.ts_writer = csv.writer(self.ts_file)
        self.ts_writer.writerow(["frame", "wall"])
        for key, topic in TOPICS.items():
            self.create_subscription(Image, topic, lambda m, k=key: self._cb(k, m), 10)
        self.create_subscription(Image, TOPICS["overhead"], self._master_cb, 10)
        self.get_logger().info(f"record_3cam → {out} (+ .timestamps.csv) @ {fps}fps")

    def _cb(self, key, msg):
        self.latest[key] = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def _master_cb(self, msg):
        # overhead 為 master：每張 overhead 觸發寫一並排幀
        ov = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.latest["overhead"] = ov
        h, w = ov.shape[:2]
        def img(k):
            im = self.latest[k]
            return im if im is not None else np.zeros((h, w, 3), np.uint8)
        combined = np.hstack([img("overhead"), img("left"), img("right")])
        if self.writer is None:
            ch, cw = combined.shape[:2]
            self.writer = cv2.VideoWriter(self.out, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (cw, ch))
        self.writer.write(combined)
        self.ts_writer.writerow([self.n, round(time.time(), 3)])
        self.ts_file.flush()
        self.n += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
        self.ts_file.close()
        self.get_logger().info(f"wrote {self.n} frames → {self.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()
    rclpy.init()
    node = ThreeCamRecorder(args.out, args.fps)

    def handler(_sig, _frame):
        node.close()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
