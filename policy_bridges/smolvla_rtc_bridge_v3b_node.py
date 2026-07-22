#!/usr/bin/env python3
"""
SmolVLA v3b RTC bridge — 16D + paper recipe（top500 dataset）。

跟 smolvla_rtc_bridge_node.py（v2 16D）的差別只有 2 個：
  1. FINETUNED_PATH 預設指向 v3b 訓練輸出
  2. CAM_TOPIC_TO_KEY 用 camera1/2/3（v3b 訓練時用 rename_map）

其他完全重用 v2 16D bridge 的 class + helpers（patch_normalizer, action16_to_ros18 etc.）

啟動：
  bash policy_bridges/run_smolvla_rtc_bridge_v3b.sh

換 checkpoint:
  bash policy_bridges/run_smolvla_rtc_bridge_v3b.sh \\
       --ckpt outputs/finetune_smolvla_v3b_top500_16d/checkpoints/006000/pretrained_model
"""

import argparse
import sys

import rclpy
from rclpy.executors import MultiThreadedExecutor

# 完整 reuse v2 bridge 的 class（包含所有 RTC + post-processing 邏輯）
# 我們只需要：1) 改 default checkpoint path  2) 改 image key mapping
import smolvla_rtc_bridge_node as v2_bridge
from smolvla_rtc_bridge_node import (
    SmolVLARTCBridgeNode,
    EMA_ALPHA, CLIP_THRESHOLD, GRIPPER_SNAP_THRESHOLD,
)


# ── v3b 特化設定 ────────────────────────────────────────────────────────────

FINETUNED_PATH_V3B = (
    "/home/teddyminelab/openarm_isaaclab/outputs/"
    "finetune_smolvla_v3b_top500_16d/checkpoints/last/pretrained_model"
)

# v3b 訓練用 rename_map，model 期望 camera1/2/3 而非 main/wrist_left/wrist_right
CAM_TOPIC_TO_KEY_V3B = {
    "/camera/overhead/image_raw": "camera1",   # top-down (was 'main')
    "/camera/left/image_raw":     "camera2",   # wrist left (was 'wrist_left')
    "/camera/right/image_raw":    "camera3",   # wrist right (was 'wrist_right')
}


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_snapshot", action="store_true")
    parser.add_argument("--log_csv", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--ckpt", type=str, default=FINETUNED_PATH_V3B,
                        help=f"checkpoint 路徑（預設 {FINETUNED_PATH_V3B}）")
    parser.add_argument("--enable_ema", action="store_true")
    parser.add_argument("--ema_alpha", type=float, default=EMA_ALPHA)
    parser.add_argument("--disable_outlier_clip", action="store_true")
    parser.add_argument("--clip_threshold", type=float, default=CLIP_THRESHOLD)
    parser.add_argument("--enable_home_return", action="store_true")
    parser.add_argument("--disable_gripper_snap", action="store_true")
    parser.add_argument("--gripper_snap_threshold", type=float, default=GRIPPER_SNAP_THRESHOLD)
    args, ros_argv = parser.parse_known_args()

    # ★★★ Monkey-patch v2 bridge 的 CAM_TOPIC_TO_KEY ★★★
    # SmolVLARTCBridgeNode.__init__ 跟 _inference_loop 都從 module-level
    # symbol 讀 CAM_TOPIC_TO_KEY，所以替換 module attr 就生效
    v2_bridge.CAM_TOPIC_TO_KEY = CAM_TOPIC_TO_KEY_V3B

    rclpy.init(args=[sys.argv[0], *ros_argv])
    node = SmolVLARTCBridgeNode(
        save_snapshot=args.save_snapshot,
        log_csv=args.log_csv,
        record_video=args.record_video,
        ckpt_path=args.ckpt,
        enable_ema=args.enable_ema,
        enable_home_return=args.enable_home_return,
        ema_alpha=args.ema_alpha,
        enable_outlier_clip=not args.disable_outlier_clip,
        clip_threshold=args.clip_threshold,
        enable_gripper_snap=not args.disable_gripper_snap,
        gripper_snap_threshold=args.gripper_snap_threshold,
    )
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_inference_thread()
        node._close_videos()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
