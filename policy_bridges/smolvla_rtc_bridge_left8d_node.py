#!/usr/bin/env python3
"""
SmolVLA + RTC bridge for OpenArm — **8D 純左臂版**（v3 finetune）

跟 smolvla_rtc_bridge_node.py 的差別：
  - Model I/O 8D（左臂 7 joint + 左 gripper）對齊 v3 訓練 dataset (basket_task_top500_left8d)
  - Right arm 8D 強制 HOME pose（跟 8D dataset distribution 一致）
  - Image keys 用 camera1/2/3（v3 用 --rename_map 訓練，model 期望 SO-100 標準命名）
  - Gripper snap 只壓 idx 7（單一左 gripper）

訂閱：
  /camera/overhead/image_raw  → observation.images.camera1   (was 'main')
  /camera/left/image_raw      → observation.images.camera2   (was 'wrist_left')
  /camera/right/image_raw     → observation.images.camera3   (was 'wrist_right')
  /Isaac/isaac_joint_states   → observation.state（抽 8D 左臂餵 model）

發布：
  /Isaac/isaac_joint_command  → 18D 目標位置
                                 ├─ 左臂 8D from model
                                 ├─ 右臂 8D = HOME constant
                                 └─ mimic finger_joint2 鏡射對應 finger_joint1

啟動：
  bash policy_bridges/run_smolvla_rtc_bridge_left8d.sh
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# 共用 ROS 18D 順序 + zero-variance normalizer patch（從 16D bridge import）
from bridge_common import ROS_JOINT_NAMES_18D
from smolvla_rtc_bridge_node import patch_normalizer_zero_variance

# ── 設定 ────────────────────────────────────────────────────────────────────

# v3 finetune checkpoint（top500_left8d, 20k steps, batch=16, paper recipe）
FINETUNED_PATH = (
    "/home/teddyminelab/openarm_isaaclab/outputs/"
    "finetune_smolvla_v3_left8d/checkpoints/last/pretrained_model"
)

# Model I/O 用 8D：只看左臂 7 個 joint + 左 gripper
MODEL_JOINT_NAMES_8D = [
    "openarm_left_joint1",  "openarm_left_joint2",  "openarm_left_joint3",  "openarm_left_joint4",
    "openarm_left_joint5",  "openarm_left_joint6",  "openarm_left_joint7",
    "openarm_left_finger_joint1",    # 左 gripper
]

# 推論時右臂 8 個 joint 強制 hold home（跟 8D dataset distribution 一致）
HOME_POSE_RIGHT_ARM_8D = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # right_joint_1~7
    0.0,                                  # right_finger_joint1 (closed)
)

# Settle 期間 publish 的 8D 左臂 HOME pose（左 gripper open 0.04）
HOME_POSE_LEFT_8D = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # left_joint_1~7
    0.04,                                 # left gripper open
)

# ★★★ v3 與 v2 的最大不同：image key 命名 ★★★
# v3 用 --rename_map 訓練（main→camera1, wrist_left→camera2, wrist_right→camera3），
# 所以 model 期望 observation.images.camera1/2/3，不能用 main/wrist_left/wrist_right。
CAM_TOPIC_TO_KEY = {
    "/camera/overhead/image_raw": "camera1",   # top-down view (was 'main')
    "/camera/left/image_raw":     "camera2",   # wrist left  (was 'wrist_left')
    "/camera/right/image_raw":    "camera3",   # wrist right (was 'wrist_right')
}

TASK_DESC      = "Pick up the cube and place it into the basket."  # ★ 必須跟 dataset tasks.parquet 完全一致
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.bfloat16
PUBLISH_HZ     = 30   # 與訓練 dataset fps 一致
SETTLE_FRAMES  = 30   # 啟動後前 1 秒 publish HOME

# RTC：lerobot doc 推薦預設（同 16D smolvla bridge）
RTC_INFERENCE_DELAY_INIT = 15
RTC_EXECUTION_HORIZON   = 10
RTC_MAX_GUIDANCE_WEIGHT = 10.0
RTC_SCHEDULE            = RTCAttentionSchedule.EXP

# ── Post-processing（policy 之後）────────────────────────────────────────
# Outlier clip（預設 ON，跟 16D 同個機制）
CLIP_THRESHOLD       = 0.15

# Gripper snap：8D 中左 gripper 在 idx 7
# v2 教訓：policy gripper output ~0.009 半閉合 → cube 滑出 → 強制 snap 到 0.0
GRIPPER_SNAP_THRESHOLD = 0.015
GRIPPER_FULL_CLOSE     = 0.000
GRIPPER_LEFT_IDX_8D    = 7

# EMA filter（opt-in，預設 OFF — outlier_clip 已夠）
EMA_ALPHA = 0.5

# Task-done detection + home ramp（opt-in）
GRIP_CLOSED_TH               = 0.005
GRIP_OPEN_TH                 = 0.030
GRIP_WIN                     = 30      # 1s @ 30fps
MIN_FRAMES_BEFORE_DONE       = 300     # 10s
HOME_RAMP_FRAMES             = 90      # 3s ramp

# ── Multi-episode benchmark ─────────────────────────────────────────────
EPISODE_FILE = "/tmp/openarm_episode.json"

SNAPSHOT_DIR = Path("/tmp/smolvla_v3_rtc_debug")
CSV_LOG_PATH = Path("/tmp/smolvla_v3_rtc_debug/inference_log.csv")
VIDEO_DIR    = Path("/tmp/smolvla_v3_rtc_debug")
VIDEO_FPS    = 30


# ── Policy 載入 ───────────────────────────────────────────────────────────

def load_policy(checkpoint_path: str, device: str, dtype: torch.dtype) -> SmolVLAPolicy:
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path, local_files_only=True)
    policy = policy.to(device=device, dtype=dtype)
    policy.eval()
    return policy


# ── 節點 ────────────────────────────────────────────────────────────────────

class SmolVLAv3RTCBridgeNode(Node):
    def __init__(self, save_snapshot: bool = False, log_csv: bool = False,
                 record_video: bool = False, ckpt_path: str = FINETUNED_PATH,
                 enable_ema: bool = False, enable_home_return: bool = False,
                 ema_alpha: float = EMA_ALPHA,
                 enable_outlier_clip: bool = True,
                 clip_threshold: float = CLIP_THRESHOLD,
                 enable_gripper_snap: bool = True,
                 gripper_snap_threshold: float = GRIPPER_SNAP_THRESHOLD):
        super().__init__("smolvla_v3_rtc_bridge_left8d")
        self.save_snapshot = save_snapshot
        self.log_csv = log_csv
        self.record_video = record_video
        self.enable_ema = enable_ema
        self.enable_home_return = enable_home_return
        self.ema_alpha = ema_alpha
        self.enable_outlier_clip = enable_outlier_clip
        self.enable_gripper_snap = enable_gripper_snap
        self.gripper_snap_threshold = gripper_snap_threshold
        self._gripper_snap_count = 0
        self.clip_threshold = clip_threshold
        self._csv_file = None
        self._video_writers: dict[str, cv2.VideoWriter] = {}
        self._video_frame_count = 0

        log = self.get_logger()
        log.info(f"Loading SmolVLA v3 (8D) from {ckpt_path}...")
        self.policy = load_policy(ckpt_path, DEVICE, DTYPE)

        # 啟用 RTC
        self.policy.config.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=RTC_EXECUTION_HORIZON,
            max_guidance_weight=RTC_MAX_GUIDANCE_WEIGHT,
            prefix_attention_schedule=RTC_SCHEDULE,
        )
        self.policy.init_rtc_processor()
        log.info(
            f"RTC enabled: execution_horizon={RTC_EXECUTION_HORIZON}, "
            f"max_guidance_weight={RTC_MAX_GUIDANCE_WEIGHT}, "
            f"schedule={RTC_SCHEDULE.value}, inference_delay (init)={RTC_INFERENCE_DELAY_INIT}"
        )

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=ckpt_path,
        )
        log.info("Patching normalizer zero-variance dims (mean/std + q01/q99):")
        patch_normalizer_zero_variance(self.preprocessor, self.postprocessor, log_fn=log.info)

        # SmolVLA 用 absolute actions（沒 use_relative_actions），不需要 re-anchor
        log.info("[INFO] SmolVLA uses absolute actions — no re-anchor needed.")

        self.action_queue = ActionQueue(self.policy.config.rtc_config)
        self._chunk_size = int(self.policy.config.chunk_size)
        log.info(f"ActionQueue ready (chunk_size={self._chunk_size})")

        # Latest observations
        self.latest_images: dict[str, torch.Tensor] = {}
        self.latest_joint_pos: dict[str, float] = {}
        self._debug_dumped = False

        # Settle phase
        self._settle_remaining = SETTLE_FRAMES

        # Inference latency EMA
        self._latency_ema_frames: float = float(RTC_INFERENCE_DELAY_INIT)

        # B 方案：EMA filter state（8D）
        self._smoothed_action: np.ndarray | None = None

        # B' 方案：outlier clip
        self._clipped_prev: np.ndarray | None = None
        self._clip_event_count = 0

        # C 方案：task-done detection + home ramp
        self._publish_frame_count = 0
        self._grip_hist: list[float] = []
        self._task_done = False
        self._home_ramp_remaining = 0
        self._home_ramp_start: np.ndarray | None = None  # 8D

        # Video writer lock
        self._video_lock = threading.Lock()

        # Multi-episode benchmark
        self._episode_lock = threading.Lock()
        self._current_episode_id: int | None = None
        self._episode_generation = 0

        # Inference thread
        self._inference_running = False
        self._inference_thread: threading.Thread | None = None
        self._inference_started = False

        # ROS callback groups
        from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
        cb_io      = ReentrantCallbackGroup()
        cb_publish = MutuallyExclusiveCallbackGroup()

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
        self.create_timer(1.0 / PUBLISH_HZ, self._publish_cb, callback_group=cb_publish)

        log.info(
            f"Post-processing flags: "
            f"EMA={'on (α=%.2f)' % self.ema_alpha if self.enable_ema else 'off'}, "
            f"outlier_clip={'on (Δ≤%.2f rad)' % self.clip_threshold if self.enable_outlier_clip else 'off'}, "
            f"gripper_snap={'on (<%.3f→0)' % self.gripper_snap_threshold if self.enable_gripper_snap else 'off'}, "
            f"home_return={'on' if self.enable_home_return else 'off'}"
        )

        # 提早啟動 inference thread（settle 期間在背景 warmup）
        self._start_inference_thread()

        log.info("SmolVLA v3 (8D) RTC bridge ready (async inference + RTC inpainting)")

    # ── Multi-episode benchmark：reset 偵測 ─────────────────────────────────

    def _check_episode_file(self):
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
        log = self.get_logger()
        with self._episode_lock:
            log.info(f"[EPISODE] Reset triggered → episode {new_episode_id}")
            self._current_episode_id = new_episode_id
            self._episode_generation += 1
            self.action_queue = ActionQueue(self.policy.config.rtc_config)
            self._settle_remaining = SETTLE_FRAMES
            self._smoothed_action = None
            self._clipped_prev = None
            self._clip_event_count = 0
            self._task_done = False
            self._home_ramp_remaining = 0
            self._home_ramp_start = None
            self._grip_hist = []
            self._publish_frame_count = 0
            self._debug_dumped = False
            self._latency_ema_frames = float(RTC_INFERENCE_DELAY_INIT)
            if hasattr(self, "_warmup_warned"):
                self._warmup_warned = False

    # ── ROS callbacks ────────────────────────────────────────────────────────

    def _image_cb(self, key: str, msg: Image):
        h, w = msg.height, msg.width
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1)[:, :, :3]
        t = torch.from_numpy(raw.copy()).permute(2, 0, 1).float() / 255.0
        self.latest_images[key] = t

    def _joint_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.latest_joint_pos[name] = pos

    # ── debug snapshot / csv / video ─────────────────────────────────────────

    def _dump_debug(self, raw_batch: dict, processed_batch: dict, action_np: np.ndarray):
        log = self.get_logger()
        log.info("=" * 60)
        log.info("[DEBUG] First inference — raw batch:")
        for k, v in raw_batch.items():
            if isinstance(v, torch.Tensor):
                log.info(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}, "
                         f"range=[{v.float().min():.3f}, {v.float().max():.3f}]")
            else:
                log.info(f"  {k}: {v!r}")
        state = raw_batch["observation.state"]
        log.info("[DEBUG] Joint state (name → value):")
        for name, val in zip(MODEL_JOINT_NAMES_8D, state.tolist()):
            log.info(f"  {name:32s} = {val:+.4f}")
        log.info(f"[DEBUG] First chunk[0,8] = {action_np.round(3).tolist()}")
        log.info("=" * 60)

        if self.save_snapshot:
            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            for cam_key in CAM_TOPIC_TO_KEY.values():
                img_t = raw_batch[f"observation.images.{cam_key}"]
                img_np = (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                PILImage.fromarray(img_np).save(SNAPSHOT_DIR / f"{cam_key}.png")
            with open(SNAPSHOT_DIR / "observation.json", "w") as f:
                json.dump({
                    "joint_state": dict(zip(MODEL_JOINT_NAMES_8D, state.tolist())),
                    "first_chunk_step0": dict(zip(MODEL_JOINT_NAMES_8D, action_np.tolist())),
                }, f, indent=2, ensure_ascii=False)
            log.info(f"[DEBUG] Snapshot saved to {SNAPSHOT_DIR}/")

    def _record_frame(self, latest_images: dict[str, torch.Tensor]):
        with self._video_lock:
            if not self._video_writers:
                VIDEO_DIR.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                for cam_key in CAM_TOPIC_TO_KEY.values():
                    if cam_key not in latest_images:
                        continue
                    _, h, w = latest_images[cam_key].shape
                    vw = cv2.VideoWriter(str(VIDEO_DIR / f"{cam_key}.mp4"), fourcc, VIDEO_FPS, (w, h))
                    if vw.isOpened():
                        self._video_writers[cam_key] = vw
                self.get_logger().info(
                    f"[VIDEO] Recording {len(self._video_writers)} cameras to {VIDEO_DIR}/*.mp4"
                )
            for cam_key, vw in self._video_writers.items():
                if cam_key not in latest_images:
                    continue
                img_t = latest_images[cam_key]
                img_np = (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                vw.write(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            self._video_frame_count += 1

    def _close_videos(self):
        for vw in self._video_writers.values():
            vw.release()
        if self._video_writers:
            self.get_logger().info(
                f"[VIDEO] Closed {len(self._video_writers)} writers, "
                f"{self._video_frame_count} frames each (~{self._video_frame_count / VIDEO_FPS:.1f} s)"
            )
        self._video_writers = {}

    def _log_frame_csv(self, state_tensor: torch.Tensor, action_np: np.ndarray):
        if self._csv_file is None:
            CSV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(CSV_LOG_PATH, "w")
            header = ["t"] + [f"state_{n}" for n in MODEL_JOINT_NAMES_8D] \
                            + [f"action_{n}" for n in MODEL_JOINT_NAMES_8D]
            self._csv_file.write(",".join(header) + "\n")
            self.get_logger().info(f"[DEBUG] CSV log started: {CSV_LOG_PATH}")
        t = self.get_clock().now().nanoseconds / 1e9
        row = [f"{t:.3f}"] + [f"{v:.6f}" for v in state_tensor.tolist()] \
                            + [f"{v:.6f}" for v in action_np]
        self._csv_file.write(",".join(row) + "\n")
        self._csv_file.flush()

    # ── 8D → 18D（右臂強制 HOME + mimic finger_joint2 = finger_joint1）─────

    def _action8_to_ros18(self, action8: np.ndarray) -> list[float]:
        """8D model output (left arm 7 joints + left gripper) → 18D Isaac command.
        右臂 8 joint 強制 HOME pose（跟 dataset distribution 一致），
        finger_joint2 鏡像對應 finger_joint1。"""
        name_to_val = dict(zip(MODEL_JOINT_NAMES_8D, action8.tolist()))
        # 右臂 hold HOME
        RIGHT_ARM_NAMES = [
            "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3", "openarm_right_joint4",
            "openarm_right_joint5", "openarm_right_joint6", "openarm_right_joint7",
            "openarm_right_finger_joint1",
        ]
        for n, v in zip(RIGHT_ARM_NAMES, HOME_POSE_RIGHT_ARM_8D):
            name_to_val[n] = v
        # mimic finger_joint2 = finger_joint1
        name_to_val["openarm_left_finger_joint2"]  = name_to_val["openarm_left_finger_joint1"]
        name_to_val["openarm_right_finger_joint2"] = name_to_val["openarm_right_finger_joint1"]
        return [name_to_val[n] for n in ROS_JOINT_NAMES_18D]

    # ── Async 推論 thread ────────────────────────────────────────────────────

    def _inference_loop(self):
        log = self.get_logger()
        log.info("[INF] Inference thread started, waiting for settle to complete.")
        cam_keys = set(CAM_TOPIC_TO_KEY.values())

        while self._inference_running and self._settle_remaining > 0:
            time.sleep(0.05)
        log.info("[INF] Settle complete, starting RTC inference.")

        while self._inference_running:
            if not cam_keys.issubset(self.latest_images):
                time.sleep(0.05); continue
            if not all(n in self.latest_joint_pos for n in MODEL_JOINT_NAMES_8D):
                time.sleep(0.05); continue
            if self._settle_remaining > 0:
                time.sleep(0.05); continue

            my_gen = self._episode_generation

            try:
                state_raw_now = torch.tensor(
                    [self.latest_joint_pos[n] for n in MODEL_JOINT_NAMES_8D], dtype=torch.float32
                )
                batch: dict = {f"observation.images.{k}": self.latest_images[k] for k in cam_keys}
                batch["observation.state"] = state_raw_now
                batch["task"] = TASK_DESC
                processed = self.preprocessor(batch)

                prev = self.action_queue.get_left_over()
                action_idx_before = self.action_queue.get_action_index()

                inference_delay_frames = max(1, int(round(self._latency_ema_frames)))

                t0 = time.time()
                with torch.no_grad(), torch.autocast(DEVICE, dtype=DTYPE):
                    chunk = self.policy.predict_action_chunk(
                        processed,
                        inference_delay=inference_delay_frames,
                        prev_chunk_left_over=prev,
                    )
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                real_delay_sec = time.time() - t0

                action_idx_after = self.action_queue.get_action_index()
                consumed_frames = max(0, action_idx_after - action_idx_before)
                real_delay_frames = consumed_frames

                if consumed_frames > 0:
                    self._latency_ema_frames = 0.3 * consumed_frames + 0.7 * self._latency_ema_frames

                chunk_processed = self.postprocessor(chunk)

                chunk_raw = chunk[0].detach() if chunk.dim() == 3 else chunk.detach()
                if isinstance(chunk_processed, torch.Tensor):
                    chunk_pp = chunk_processed[0].detach() if chunk_processed.dim() == 3 \
                               else chunk_processed.detach()
                else:
                    chunk_pp = torch.as_tensor(chunk_processed)

                # ★ 8D：切 [:, :8] 而非 16D 的 [:, :16]
                chunk_pp_8 = chunk_pp[:, :8].float()

                if not self._debug_dumped:
                    first = chunk_pp_8[0].cpu().numpy()
                    self._dump_debug(batch, processed, first)
                    self._debug_dumped = True
                    log.info(
                        f"[INF] First chunk: wallclock={real_delay_sec*1000:.0f}ms, "
                        f"queue consumed={consumed_frames} frames, "
                        f"inference_delay used={inference_delay_frames}"
                    )

                with self._episode_lock:
                    if my_gen == self._episode_generation:
                        self.action_queue.merge(
                            chunk_raw,
                            chunk_pp_8,
                            real_delay_frames,
                            action_idx_before,
                        )
                    else:
                        log.info("[INF] Discarding stale chunk from previous episode generation")

            except Exception as e:
                log.error(f"[INF] Inference loop error: {e!r}")
                time.sleep(0.1)

        log.info("[INF] Inference thread exiting.")

    def _start_inference_thread(self):
        if not self._inference_started:
            self._inference_running = True
            self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self._inference_thread.start()
            self._inference_started = True

    def stop_inference_thread(self):
        self._inference_running = False
        if self._inference_thread:
            self._inference_thread.join(timeout=2.0)

    # ── 後處理 helper ────────────────────────────────────────────────────────

    def _apply_outlier_clip(self, raw_action: np.ndarray) -> np.ndarray:
        if self._clipped_prev is None:
            self._clipped_prev = raw_action.copy()
            return raw_action
        delta = raw_action - self._clipped_prev
        max_delta = self.clip_threshold
        clipped_delta = np.clip(delta, -max_delta, max_delta)
        if np.any(np.abs(delta) > max_delta):
            self._clip_event_count += 1
        out = self._clipped_prev + clipped_delta
        self._clipped_prev = out
        return out

    def _apply_ema(self, raw_action: np.ndarray) -> np.ndarray:
        if self._smoothed_action is None:
            self._smoothed_action = raw_action.copy()
            return raw_action
        smoothed = self.ema_alpha * raw_action + (1.0 - self.ema_alpha) * self._smoothed_action
        self._smoothed_action = smoothed
        return smoothed

    def _check_task_done(self):
        """8D 版：只看左 gripper（idx 7）"""
        if not self.enable_home_return:
            return
        if self._task_done:
            return
        if self._publish_frame_count < MIN_FRAMES_BEFORE_DONE:
            return
        if "openarm_left_finger_joint1" not in self.latest_joint_pos:
            return

        grip = self.latest_joint_pos["openarm_left_finger_joint1"]
        self._grip_hist.append(grip)
        if len(self._grip_hist) > GRIP_WIN:
            self._grip_hist.pop(0)
        if len(self._grip_hist) < GRIP_WIN:
            return

        win = self._grip_hist
        had_closed   = min(win) < GRIP_CLOSED_TH
        recent_open  = min(win[-GRIP_WIN // 2:]) > GRIP_OPEN_TH
        if had_closed and recent_open:
            self._task_done = True
            self._home_ramp_remaining = HOME_RAMP_FRAMES
            self._home_ramp_start = np.asarray(
                [self.latest_joint_pos.get(n, 0.0) for n in MODEL_JOINT_NAMES_8D],
                dtype=np.float64,
            )
            self.get_logger().info(
                f"[TASK-DONE] gripper open→closed→open detected at frame "
                f"{self._publish_frame_count}. Starting {HOME_RAMP_FRAMES}-frame home ramp."
            )

    def _home_ramp_action(self) -> np.ndarray:
        """從 _home_ramp_start (8D) 線性 ramp 到 HOME_POSE_LEFT_8D"""
        total = HOME_RAMP_FRAMES
        elapsed = total - self._home_ramp_remaining + 1
        alpha = min(1.0, elapsed / total)
        start = self._home_ramp_start if self._home_ramp_start is not None else np.zeros(8)
        target = np.asarray(HOME_POSE_LEFT_8D, dtype=np.float64)
        self._home_ramp_remaining = max(0, self._home_ramp_remaining - 1)
        return (1.0 - alpha) * start + alpha * target

    # ── Publish timer（30Hz）─────────────────────────────────────────────────

    def _publish_cb(self):
        self._check_episode_file()

        # ── Settle phase ──
        if self._settle_remaining > 0:
            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name         = ROS_JOINT_NAMES_18D
            cmd.position     = self._action8_to_ros18(np.asarray(HOME_POSE_LEFT_8D, dtype=np.float32))
            self.cmd_pub.publish(cmd)
            self._settle_remaining -= 1
            if self._settle_remaining == 0:
                self.get_logger().info(
                    f"Settle complete. Action queue size: {self.action_queue.qsize()} (warmed during settle)."
                )
            return

        # ── 任務完成後 → home ramp ──
        if self._task_done:
            if self._home_ramp_remaining > 0:
                action_np = self._home_ramp_action().astype(np.float32)
            else:
                action_np = np.asarray(HOME_POSE_LEFT_8D, dtype=np.float32)

            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name         = ROS_JOINT_NAMES_18D
            cmd.position     = self._action8_to_ros18(action_np)
            self.cmd_pub.publish(cmd)
            self._publish_frame_count += 1

            if self.log_csv and all(n in self.latest_joint_pos for n in MODEL_JOINT_NAMES_8D):
                state_now = torch.tensor(
                    [self.latest_joint_pos[n] for n in MODEL_JOINT_NAMES_8D], dtype=torch.float32
                )
                self._log_frame_csv(state_now, action_np)
            if self.record_video:
                self._record_frame(self.latest_images)
            return

        # ── 一般 policy 模式 ──
        action = self.action_queue.get()
        if action is None:
            if not getattr(self, "_warmup_warned", False):
                self.get_logger().info("Waiting for first RTC chunk...")
                self._warmup_warned = True
            return
        action_np = action.detach().cpu().numpy()
        if action_np.shape[0] != 8:
            action_np = action_np[:8]

        # B' 方案：outlier clip
        if self.enable_outlier_clip:
            action_np = self._apply_outlier_clip(action_np)

        # Gripper snap（8D 中只有 idx 7）
        if self.enable_gripper_snap:
            if action_np[GRIPPER_LEFT_IDX_8D] < self.gripper_snap_threshold:
                if action_np[GRIPPER_LEFT_IDX_8D] > GRIPPER_FULL_CLOSE:
                    self._gripper_snap_count += 1
                action_np[GRIPPER_LEFT_IDX_8D] = GRIPPER_FULL_CLOSE

        # B 方案：EMA filter
        if self.enable_ema:
            action_np = self._apply_ema(action_np)

        # C 方案：task-done detection
        self._publish_frame_count += 1
        self._check_task_done()

        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name         = ROS_JOINT_NAMES_18D
        cmd.position     = self._action8_to_ros18(action_np)
        self.cmd_pub.publish(cmd)

        if self.log_csv and all(n in self.latest_joint_pos for n in MODEL_JOINT_NAMES_8D):
            state_now = torch.tensor(
                [self.latest_joint_pos[n] for n in MODEL_JOINT_NAMES_8D], dtype=torch.float32
            )
            self._log_frame_csv(state_now, action_np)
        if self.record_video:
            self._record_frame(self.latest_images)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_snapshot", action="store_true",
                        help="第一次推論時把影像 + state + chunk[0] 存到 /tmp/smolvla_v3_rtc_debug/")
    parser.add_argument("--log_csv", action="store_true",
                        help="每幀 state + action 寫到 /tmp/smolvla_v3_rtc_debug/inference_log.csv")
    parser.add_argument("--record_video", action="store_true",
                        help="每幀錄三個 camera 到 /tmp/smolvla_v3_rtc_debug/{camera1,camera2,camera3}.mp4")
    parser.add_argument("--ckpt", type=str, default=FINETUNED_PATH,
                        help=f"checkpoint 路徑（預設 {FINETUNED_PATH}）。"
                             "也可指向中間 checkpoint 比較不同 step 的表現，e.g. checkpoints/006000/pretrained_model")
    parser.add_argument("--enable_ema", action="store_true",
                        help="開啟 publish 端 EMA filter")
    parser.add_argument("--ema_alpha", type=float, default=EMA_ALPHA)
    parser.add_argument("--disable_outlier_clip", action="store_true",
                        help="關掉 outlier clip（預設 ON）")
    parser.add_argument("--clip_threshold", type=float, default=CLIP_THRESHOLD)
    parser.add_argument("--enable_home_return", action="store_true",
                        help="開啟任務完成偵測 + ramp 回 home")
    parser.add_argument("--disable_gripper_snap", action="store_true",
                        help=f"關掉 gripper snap（預設 ON：policy gripper < {GRIPPER_SNAP_THRESHOLD} 時 snap 到 0.0）")
    parser.add_argument("--gripper_snap_threshold", type=float, default=GRIPPER_SNAP_THRESHOLD)
    args, ros_argv = parser.parse_known_args()

    rclpy.init(args=[sys.argv[0], *ros_argv])
    node = SmolVLAv3RTCBridgeNode(
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
    from rclpy.executors import MultiThreadedExecutor
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
