#!/usr/bin/env python3
"""
ACT (finetuned) bridge node for OpenArm bimanual pick-and-place.

與 pi05_bridge_node.py 共享 ROS topic 與 16D ↔ 18D mimic 轉換，
差別：
  - 載入 ACT（~80M）→ 直接 from_pretrained，不用 meta-device hack
  - 開啟 temporal ensembling（n_action_steps=1）→ 每幀 query 一次 select_action()
    → 不需要 chunk buffer / LOW_WATERMARK 邏輯
  - ACT 不吃 language → batch 不放 task

訂閱：
  /camera/overhead/image_raw  → observation.images.main
  /camera/left/image_raw      → observation.images.wrist_left
  /camera/right/image_raw     → observation.images.wrist_right
  /Isaac/isaac_joint_states   → observation.state（投影成 16D folding 順序餵 model）

發布：
  /Isaac/isaac_joint_command  → 18D 目標位置（model 16D 輸出 + mimic finger_joint2 鏡射）

啟動：
  bash policy_bridges/run_act_bridge.sh
"""

import argparse
import json
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

# ── 設定 ────────────────────────────────────────────────────────────────────

FINETUNED_PATH = "/home/teddyminelab/openarm_isaaclab/outputs/finetune_act_basket/checkpoints/last/pretrained_model"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
PUBLISH_HZ     = 30   # 與訓練 dataset fps 一致；temporal ensembling 要求每幀 query
SETTLE_FRAMES  = 30   # 啟動後前 1 秒 publish 當前 state，讓 PD 穩定

# 16D folding 順序：[R1..R7, R_grip, L1..L7, L_grip]
# 與 collect_basket_demos.py / pi05_bridge_node.py / notebook 完全一致
MODEL_JOINT_NAMES_16D = [
    "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3", "openarm_right_joint4",
    "openarm_right_joint5", "openarm_right_joint6", "openarm_right_joint7",
    "openarm_right_finger_joint1",
    "openarm_left_joint1",  "openarm_left_joint2",  "openarm_left_joint3",  "openarm_left_joint4",
    "openarm_left_joint5",  "openarm_left_joint6",  "openarm_left_joint7",
    "openarm_left_finger_joint1",
]

# ROS publish 順序（articulation 18D，含 mimic finger_joint2）
ROS_JOINT_NAMES_18D = [
    "openarm_left_joint1",         "openarm_right_joint1",
    "openarm_left_joint2",         "openarm_right_joint2",
    "openarm_left_joint3",         "openarm_right_joint3",
    "openarm_left_joint4",         "openarm_right_joint4",
    "openarm_left_joint5",         "openarm_right_joint5",
    "openarm_left_joint6",         "openarm_right_joint6",
    "openarm_left_joint7",         "openarm_right_joint7",
    "openarm_left_finger_joint1",  "openarm_left_finger_joint2",
    "openarm_right_finger_joint1", "openarm_right_finger_joint2",
]

CAM_TOPIC_TO_KEY = {
    "/camera/overhead/image_raw": "main",
    "/camera/left/image_raw":     "wrist_left",
    "/camera/right/image_raw":    "wrist_right",
}

SNAPSHOT_DIR = Path("/tmp/act_debug")
CSV_LOG_PATH = Path("/tmp/act_debug/inference_log.csv")
VIDEO_DIR    = Path("/tmp/act_debug")
VIDEO_FPS    = 30   # 對齊 PUBLISH_HZ；播放時跟訓練 dataset 同 fps

# Multi-episode benchmark：透過 /tmp/openarm_episode.json 跟 Isaac scene 同步
EPISODE_FILE = "/tmp/openarm_episode.json"

# Settle 期間發給 ArticulationCtrl 的目標 pose（跟 Isaac reset_episode 寫的 home 一致）
# 16D folding 順序 [R1..R7, R_grip, L1..L7, L_grip]
HOME_POSE_16D = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04,
)


# ── 節點 ────────────────────────────────────────────────────────────────────

class ACTBridgeNode(Node):
    def __init__(self, save_snapshot: bool = False, log_csv: bool = False,
                 record_video: bool = False, ckpt_path: str = FINETUNED_PATH):
        super().__init__("act_bridge")
        self.save_snapshot = save_snapshot
        self.log_csv = log_csv
        self.record_video = record_video
        self._csv_file = None
        self._video_writers: dict[str, cv2.VideoWriter] = {}
        self._video_frame_count = 0

        log = self.get_logger()
        log.info(f"Loading ACT model from {ckpt_path}...")
        self.policy = ACTPolicy.from_pretrained(ckpt_path)
        self.policy.to(DEVICE)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=ckpt_path,
        )

        # 重置 temporal ensembler（episode 開始時必做）
        self.policy.reset()
        log.info(
            f"ACT ready: chunk_size={self.policy.config.chunk_size}, "
            f"temporal_ensemble_coeff={self.policy.config.temporal_ensemble_coeff}, "
            f"device={DEVICE}"
        )

        # Latest observations
        self.latest_images: dict[str, torch.Tensor] = {}
        self.latest_joint_pos: dict[str, float] = {}
        self._infer_lock = threading.Lock()
        self._debug_dumped = False

        # Settle phase：開頭 SETTLE_FRAMES 幀 publish 當前 state
        self._settle_remaining = SETTLE_FRAMES

        # Multi-episode benchmark：reset 同步
        self._current_episode_id: int | None = None

        # Callback groups
        from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
        cb_io     = ReentrantCallbackGroup()
        cb_infer  = MutuallyExclusiveCallbackGroup()  # 推論互斥

        for topic, key in CAM_TOPIC_TO_KEY.items():
            self.create_subscription(
                Image, topic,
                lambda msg, k=key: self._image_cb(k, msg),
                10, callback_group=cb_io,
            )
        self.create_subscription(
            JointState, "/Isaac/isaac_joint_states", self._joint_cb, 10,
            callback_group=cb_io,
        )

        self.cmd_pub = self.create_publisher(JointState, "/Isaac/isaac_joint_command", 10)

        # 推論 + publish 同一個 timer：temporal ensembling 要求每幀 query
        self.create_timer(1.0 / PUBLISH_HZ, self._step_cb, callback_group=cb_infer)

        log.info("ACT bridge ready (temporal ensembling on, query every step @ 30Hz)")

    # ── Multi-episode benchmark：reset 偵測 ─────────────────────────────────

    def _check_episode_file(self):
        """Poll EPISODE_FILE，發現新 episode id（resetting 或 ready）就觸發 reset。"""
        try:
            with open(EPISODE_FILE, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        ep_id = data.get("episode")
        state = data.get("state")
        if ep_id is None or state not in ("resetting", "ready"):
            return
        if self._current_episode_id == ep_id:
            return
        self._do_episode_reset(ep_id)

    def _do_episode_reset(self, new_episode_id: int):
        """重設 ACT temporal ensembler + settle counter，為新 episode 做準備。"""
        log = self.get_logger()
        log.info(f"[EPISODE] Reset triggered → episode {new_episode_id}")
        self._current_episode_id = new_episode_id
        self.policy.reset()              # 清空 ACT temporal ensembling buffer
        self._settle_remaining = SETTLE_FRAMES
        self._debug_dumped = False

    # ── callbacks ────────────────────────────────────────────────────────────

    def _image_cb(self, key: str, msg: Image):
        h, w = msg.height, msg.width
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1)[:, :, :3]
        t = torch.from_numpy(raw.copy()).permute(2, 0, 1).float() / 255.0
        self.latest_images[key] = t

    def _joint_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.latest_joint_pos[name] = pos

    # ── debug snapshot（首次推論時 dump 一次完整輸入）──────────────────────────

    def _dump_debug(self, raw_batch: dict, processed_batch: dict, action_np: np.ndarray):
        log = self.get_logger()
        log.info("=" * 60)
        log.info("[DEBUG] First inference — raw batch:")
        for k, v in raw_batch.items():
            if isinstance(v, torch.Tensor):
                log.info(
                    f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}, "
                    f"range=[{v.float().min():.3f}, {v.float().max():.3f}]"
                )
            else:
                log.info(f"  {k}: {v!r}")

        state = raw_batch["observation.state"]
        log.info("[DEBUG] Joint state (name → value):")
        for name, val in zip(MODEL_JOINT_NAMES_16D, state.tolist()):
            log.info(f"  {name:32s} = {val:+.4f}")

        log.info(f"[DEBUG] Ensembled action[16] = {action_np.round(3).tolist()}")
        log.info(f"[DEBUG] action range: min={action_np.min():.3f}, max={action_np.max():.3f}")
        log.info("=" * 60)

        if self.save_snapshot:
            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            for cam_key in CAM_TOPIC_TO_KEY.values():
                img_t = raw_batch[f"observation.images.{cam_key}"]
                img_np = (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                PILImage.fromarray(img_np).save(SNAPSHOT_DIR / f"{cam_key}.png")
            snapshot = {
                "joint_state":          dict(zip(MODEL_JOINT_NAMES_16D, state.tolist())),
                "first_action_ensembled": dict(zip(MODEL_JOINT_NAMES_16D, action_np.tolist())),
            }
            with open(SNAPSHOT_DIR / "observation.json", "w") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)
            log.info(f"[DEBUG] Snapshot saved to {SNAPSHOT_DIR}/")

    # ── 每幀錄影（mp4 per camera）──────────────────────────────────────────────

    def _record_frame(self, raw_batch: dict):
        """每幀寫 3 個 mp4（main / wrist_left / wrist_right）。Lazy init writers."""
        if not self._video_writers:
            VIDEO_DIR.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            for cam_key in CAM_TOPIC_TO_KEY.values():
                img_t = raw_batch[f"observation.images.{cam_key}"]
                _, h, w = img_t.shape
                path = str(VIDEO_DIR / f"{cam_key}.mp4")
                vw = cv2.VideoWriter(path, fourcc, VIDEO_FPS, (w, h))
                if not vw.isOpened():
                    self.get_logger().error(f"[VIDEO] Failed to open writer for {path}")
                    continue
                self._video_writers[cam_key] = vw
            self.get_logger().info(
                f"[VIDEO] Recording {len(self._video_writers)} cameras to {VIDEO_DIR}/*.mp4 @ {VIDEO_FPS} fps"
            )

        for cam_key, vw in self._video_writers.items():
            img_t = raw_batch[f"observation.images.{cam_key}"]
            # [3, H, W] in [0,1] → [H, W, 3] uint8 BGR (cv2 順序)
            img_np = (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            vw.write(img_bgr)
        self._video_frame_count += 1

    def _close_videos(self):
        for cam_key, vw in self._video_writers.items():
            vw.release()
        if self._video_writers:
            self.get_logger().info(
                f"[VIDEO] Released {len(self._video_writers)} writers, {self._video_frame_count} frames each "
                f"(~{self._video_frame_count / VIDEO_FPS:.1f} s)"
            )
        self._video_writers = {}

    # ── 每幀 CSV log ──────────────────────────────────────────────────────────

    def _log_frame_csv(self, state_tensor: torch.Tensor, action_np: np.ndarray):
        if self._csv_file is None:
            CSV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(CSV_LOG_PATH, "w")
            header = ["t"] + [f"state_{n}" for n in MODEL_JOINT_NAMES_16D] \
                            + [f"action_{n}" for n in MODEL_JOINT_NAMES_16D]
            self._csv_file.write(",".join(header) + "\n")
            self.get_logger().info(f"[DEBUG] CSV log started: {CSV_LOG_PATH}")

        t = self.get_clock().now().nanoseconds / 1e9
        state_vals = state_tensor.tolist()
        row = [f"{t:.3f}"] + [f"{v:.6f}" for v in state_vals] + [f"{v:.6f}" for v in action_np]
        self._csv_file.write(",".join(row) + "\n")
        self._csv_file.flush()

    # ── 16D → 18D（mimic finger_joint2 = finger_joint1）──────────────────────

    def _action16_to_ros18(self, action16: np.ndarray) -> list[float]:
        name_to_val = dict(zip(MODEL_JOINT_NAMES_16D, action16.tolist()))
        name_to_val["openarm_left_finger_joint2"]  = name_to_val["openarm_left_finger_joint1"]
        name_to_val["openarm_right_finger_joint2"] = name_to_val["openarm_right_finger_joint1"]
        return [name_to_val[n] for n in ROS_JOINT_NAMES_18D]

    # ── 主迴圈：每幀 query 一次 ACT，溫度集成由 policy.select_action 內部處理 ──

    def _step_cb(self):
        # 每幀檢查 episode reset signal（cheap：只讀小 JSON）
        self._check_episode_file()

        # Settle phase：永遠 publish HOME pose（避免蓋掉 Isaac reset 的 home target）
        if self._settle_remaining > 0:
            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name         = ROS_JOINT_NAMES_18D
            cmd.position     = self._action16_to_ros18(np.asarray(HOME_POSE_16D, dtype=np.float32))
            self.cmd_pub.publish(cmd)
            self._settle_remaining -= 1
            if self._settle_remaining == 0:
                self.get_logger().info("Settle phase complete, switching to ACT output.")
            return

        cam_keys = set(CAM_TOPIC_TO_KEY.values())
        if not cam_keys.issubset(self.latest_images):
            return
        if not all(n in self.latest_joint_pos for n in MODEL_JOINT_NAMES_16D):
            return

        if not self._infer_lock.acquire(blocking=False):
            # 上一幀推論還沒結束（罕見，ACT ~10ms 應該追得上 30Hz）
            return

        try:
            # 組 batch（不放 task——ACT 不吃 language）
            batch: dict = {f"observation.images.{k}": self.latest_images[k] for k in cam_keys}
            batch["observation.state"] = torch.tensor(
                [self.latest_joint_pos[n] for n in MODEL_JOINT_NAMES_16D], dtype=torch.float32
            )

            processed = self.preprocessor(batch)

            # select_action：ACT 內部跑 chunk + temporal ensembler，回傳當前幀的 ensembled action
            with torch.inference_mode():
                action = self.policy.select_action(processed)

            action = self.postprocessor(action)
            action_np = action.detach().float().cpu().numpy()
            if action_np.ndim == 2:
                action_np = action_np[0]
            action_np = action_np[:16]

            if not self._debug_dumped:
                self._dump_debug(batch, processed, action_np)
                self._debug_dumped = True

            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name         = ROS_JOINT_NAMES_18D
            cmd.position     = self._action16_to_ros18(action_np)
            self.cmd_pub.publish(cmd)

            if self.log_csv:
                self._log_frame_csv(batch["observation.state"], action_np)
            if self.record_video:
                self._record_frame(batch)
        finally:
            self._infer_lock.release()


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_snapshot", action="store_true",
        help="第一次推論時，把三張影像 + state + action 存到 /tmp/act_debug/"
    )
    parser.add_argument(
        "--log_csv", action="store_true",
        help="每幀的 state + ensembled action 寫到 /tmp/act_debug/inference_log.csv"
    )
    parser.add_argument(
        "--record_video", action="store_true",
        help="每幀錄三個 camera 到 /tmp/act_debug/{main,wrist_left,wrist_right}.mp4 (~3-8 MB/min)"
    )
    parser.add_argument(
        "--ckpt", type=str, default=FINETUNED_PATH,
        help="checkpoint 路徑（預設 outputs/finetune_act_basket/checkpoints/last/pretrained_model）"
    )
    args, ros_argv = parser.parse_known_args()

    rclpy.init(args=[sys.argv[0], *ros_argv])
    node = ACTBridgeNode(
        save_snapshot=args.save_snapshot, log_csv=args.log_csv,
        record_video=args.record_video, ckpt_path=args.ckpt,
    )
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._close_videos()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
