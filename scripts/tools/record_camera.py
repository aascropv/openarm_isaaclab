#!/usr/bin/env python3
"""錄 ROS2 camera image topic → mp4。跑到收到 SIGTERM/SIGINT 為止（會 flush 檔案）。

用法（在 ROS2 環境下）：
  python3 scripts/tools/record_camera.py --topic /camera/overhead/image_raw --out /tmp/ep.mp4
"""
import argparse
import signal
import sys

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class Recorder(Node):
    def __init__(self, topic: str, out: str, fps: float):
        super().__init__("camera_recorder")
        self.bridge = CvBridge()
        self.writer = None
        self.out = out
        self.fps = fps
        self.n = 0
        self.create_subscription(Image, topic, self.cb, 10)
        self.get_logger().info(f"recording {topic} -> {out} @ {fps}fps")

    def cb(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = cv2.VideoWriter(
                self.out, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self.writer.write(img)
        self.n += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
        self.get_logger().info(f"wrote {self.n} frames -> {self.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/camera/overhead/image_raw")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    rclpy.init()
    node = Recorder(args.topic, args.out, args.fps)

    def handler(_sig, _frame):
        node.close()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
