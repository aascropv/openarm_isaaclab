#!/usr/bin/env python3
"""
SmolVLA + RTC (Real-Time Chunking) bridge node for OpenArm bimanual pick-and-place.

跟 pi05_rtc_bridge_node.py 的差別：
  - 換成 SmolVLAPolicy（~700M，4× 小於 PI0.5 3B）
  - SmolVLA 沒有 use_relative_actions → 拿掉所有 B 方案 re-anchor 邏輯（absolute actions 不需要）
  - SmolVLA 用 MEAN_STD normalize（不是 PI0.5 的 QUANTILES）→ patch 函式擴充 std 維度
  - chunk_size=50（PI0.5 是 30）→ RTC execution_horizon 仍用 10（doc 推薦值，與 chunk_size 無關）

訂閱 / 發布 topic 與 PI0.5 bridge 完全一致：
  /camera/{overhead,left,right}/image_raw → observation.images.{main,wrist_left,wrist_right}
  /Isaac/isaac_joint_states               → observation.state（16D folding 順序）
  /Isaac/isaac_joint_command              → 18D 目標位置（model 16D + mimic finger_joint2 鏡射）

啟動：
  bash policy_bridges/run_smolvla_rtc_bridge.sh
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

# 共用 joint / 相機常數（單臂任務但仍走 16D folding 格式）
from bridge_common import (
    MODEL_JOINT_NAMES_16D,
    ROS_JOINT_NAMES_18D,
    CAM_TOPIC_TO_KEY,
)

# ── 設定 ────────────────────────────────────────────────────────────────────

FINETUNED_PATH = (
    "/home/teddyminelab/openarm_isaaclab/outputs/"
    "finetune_smolvla_basket/checkpoints/last/pretrained_model"
)
TASK_DESC      = "Pick up the cube and place it into the basket."  # ★ 必須跟 dataset tasks.parquet 完全一致；改字串會把 policy 推離訓練分佈
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.bfloat16
PUBLISH_HZ     = 30   # 與訓練 dataset fps 一致
SETTLE_FRAMES  = 30   # 啟動後前 1 秒 publish 當前 state，讓 PD 穩定

# RTC：lerobot doc 推薦預設。SmolVLA 比 PI0.5 小 4×，inference 應更快
#  - 首次 inference_delay 用保守估計 15 frames（標稱 RTX 3090 量到 ~140ms ≈ 4 frames，
#    但 Isaac Sim 搶 GPU 會放大）後續 EMA 自動追蹤實測
#  - execution_horizon=10：doc 推薦值（chunk_size=50 → 仍留 40 frames buffer）
#  - max_guidance_weight=10：doc 推薦值
#  - EXP schedule：doc "recommended for getting started"
RTC_INFERENCE_DELAY_INIT = 15
RTC_EXECUTION_HORIZON   = 10
RTC_MAX_GUIDANCE_WEIGHT = 10.0
RTC_SCHEDULE            = RTCAttentionSchedule.EXP

MIN_NORM_RANGE = 0.01  # 與 patch_basket_v2_stats.py / pi05 bridge 同步

# ── Action smoothing & home-return（後處理，policy 之外）────────────────────
# B 方案：在 publish 端對 16D action 做 EMA low-pass。
#   smoothed[t] = α * new[t] + (1-α) * smoothed[t-1]
#   α=0.5 → -3dB cutoff ≈ 4.8 Hz @ 30 fps，明顯降 0.014 rad/frame jitter
#   注意：EMA 會造成 1-2 frame timing lag，跟 gripper clamp 時序衝突
EMA_ALPHA            = 0.5
GRIPPER_DIMS         = (7, 15)  # right_finger / left_finger 在 16D 順序中的 index

# B' 方案：Outlier clip — 偵測單幀 velocity 大跳，限制 max |Δaction|。
#   只壓 outlier，不像 EMA 全程降頻，所以不會造成正常動作 lag。
#   閾值 0.15 rad/frame ≈ 4.5 rad/s = 258°/s，遠大於任何合理機器人動作速度，
#   但小於 RTC chunk-seam crash 的 0.4-0.8 rad/frame 異常跳。
CLIP_THRESHOLD       = 0.15

# Gripper override：teleop 收集時 close 只到 ~0.009（不夠緊），cube transport 期間
# 會滑出。policy gripper output < GRIPPER_SNAP_THRESHOLD 時直接 snap 到 0.0（全閉合）。
# 16D action idx 7=right gripper, idx 15=left gripper。
# 閾值 0.015：訓練資料 transport gripper 在 [0.007, 0.010]，policy 推論時若還在 approach
# 階段可能短暫降到 0.020 但不該被 snap；0.015 是「明確 commit close」的安全閾值。
# （0.025 太鬆，會誤觸發 approach 期間的 transient close intent，造成 cube 上方就先關夾爪）
GRIPPER_SNAP_THRESHOLD = 0.015
GRIPPER_FULL_CLOSE     = 0.000
GRIPPER_RIGHT_IDX      = 7
GRIPPER_LEFT_IDX       = 15

# C 方案：偵測任務完成 → bypass policy 用 ramp 回 home。
#   觸發條件：過去 GRIP_WIN frames 內 left gripper 從 closed (<GRIP_CLOSED_TH)
#            → opened (>GRIP_OPEN_TH)，且 inference 已執行至少 MIN_FRAMES_BEFORE_DONE 幀
#            （避免 settle 後第一個 chunk 抖動時誤觸發）。
#   觸發後 lock，從當前 state 線性 ramp 到 HOME_POSE，總共 HOME_RAMP_FRAMES。
GRIP_CLOSED_TH               = 0.005
GRIP_OPEN_TH                 = 0.030
GRIP_WIN                     = 30      # 1s @ 30fps
MIN_FRAMES_BEFORE_DONE       = 300     # 10s 後才開始考慮 task-done
HOME_RAMP_FRAMES             = 90      # 3s ramp（夠慢避免 PD 暴衝）
HOME_POSE_16D                = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04,  # right arm + open gripper
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04,  # left arm + open gripper
)

# D 方案（取代 C 的 naive ramp）：scripted return —— 偵測目標水果穩定入目標籃 →
# 播 demo 回程軌跡（dense joint path，高抬繞行、不掃籃子）。前 RETURN_BLEND_FRAMES 幀
# 用 offset-decay 從當前姿態平滑接上 demo 軌跡（吸收閉環 place pose 的小偏差）。
RETURN_BLEND_FRAMES  = 30
# ★ 握持門檻用 0.020（非 C 方案的 0.005）：eval --disable_gripper_snap 下實際握持 ~0.007-0.011
#   （demo idx7 中位 ~0.01），release 前 1s 窗口最低就是握持值 ~0.01，用 0.005 會永遠不觸發。
PLACE_GRIP_CLOSED_TH = 0.020   # 曾握持（<此值）
PLACE_GRIP_OPEN_TH   = 0.030   # release（>此值，open≈0.04）

# ── Multi-episode benchmark：透過 /tmp/openarm_episode.json 跟 Isaac scene 同步 ──
# Isaac 在 reset 完成時寫入 {"state": "ready", "episode": N, "cube": [x,y,z]}。
# Bridge 偵測 episode N 變化 → 清空 ActionQueue + smoothing state + 重設 settle，
# 確保 policy 從乾淨 state 開始下個 episode。
EPISODE_FILE = "/tmp/openarm_episode.json"

SNAPSHOT_DIR = Path("/tmp/smolvla_rtc_debug")
CSV_LOG_PATH = Path("/tmp/smolvla_rtc_debug/inference_log.csv")
VIDEO_DIR    = Path("/tmp/smolvla_rtc_debug")
VIDEO_FPS    = 30


# ── Policy 載入 ───────────────────────────────────────────────────────────
#
# SmolVLA (~700M) 比 PI0.5 (3B) 小 4×，記憶體不是問題，直接用標準 from_pretrained。
# （PI0.5 bridge 用 meta-device + safe_open 是為了壓 28GB CPU 峰值；SmolVLA 不需要）
# SmolVLMWithExpertModel.__init__ 內部會 AutoModelForImageTextToText.from_pretrained
# 載 SmolVLM2 backbone weights（不允許 meta device），所以這條路也不能用 meta trick。

def load_policy(checkpoint_path: str, device: str, dtype: torch.dtype) -> SmolVLAPolicy:
    policy = SmolVLAPolicy.from_pretrained(checkpoint_path, local_files_only=True)
    policy = policy.to(device=device, dtype=dtype)
    policy.eval()
    return policy


def patch_normalizer_zero_variance(preprocessor, postprocessor, log_fn=print):
    """
    擴張 zero-variance 維度，避免 normalize 時除以微小值爆炸。

    SmolVLA 用 MEAN_STD（state/action），所以這版同時處理 q01/q99 與 mean/std。
    （PI0.5 bridge 的版本只處理 q01/q99。即使訓練前已 patch dataset stats.json，
    runtime 仍 patch 一次作 defense in depth。）
    """
    TARGET_FEATS = ("observation.state", "action")

    def _patch_step(step, prefix):
        for feat_key in TARGET_FEATS:
            # 1. dict (numpy ndarray) — 不一定被 normalize 用到，但保持一致
            if hasattr(step, "stats") and isinstance(step.stats, dict) and feat_key in step.stats:
                s = step.stats[feat_key]
                if isinstance(s, dict):
                    if "q01" in s and "q99" in s and isinstance(s["q01"], np.ndarray):
                        q01, q99 = s["q01"], s["q99"]
                        diff = q99 - q01
                        small = np.abs(diff) < MIN_NORM_RANGE
                        if small.any():
                            half = MIN_NORM_RANGE / 2
                            mid = (q01 + q99) / 2
                            s["q01"] = np.where(small, mid - half, q01).astype(q01.dtype)
                            s["q99"] = np.where(small, mid + half, q99).astype(q99.dtype)
                    if "std" in s and isinstance(s["std"], np.ndarray):
                        st = s["std"]
                        small = np.abs(st) < MIN_NORM_RANGE / 2
                        if small.any():
                            s["std"] = np.where(small, MIN_NORM_RANGE / 2, st).astype(st.dtype)

            # 2. _tensor_stats（normalize forward 真正用這個）
            tstats = getattr(step, "_tensor_stats", None)
            if tstats is None or feat_key not in tstats:
                continue
            ts = tstats[feat_key]
            if not isinstance(ts, dict):
                continue

            # q01 / q99
            if "q01" in ts and "q99" in ts:
                q01_t, q99_t = ts["q01"], ts["q99"]
                diff = q99_t - q01_t
                small = diff.abs() < MIN_NORM_RANGE
                n = int(small.sum().item())
                if n > 0:
                    half = MIN_NORM_RANGE / 2
                    mid = (q01_t + q99_t) / 2
                    ts["q01"] = torch.where(small, mid - half, q01_t).to(q01_t.dtype)
                    ts["q99"] = torch.where(small, mid + half, q99_t).to(q99_t.dtype)
                    log_fn(f"  [{prefix}] {feat_key}: q01/q99 擴張 {n} 維 (denom ≥ {MIN_NORM_RANGE})")

            # mean / std（SmolVLA 主要走這條）
            if "std" in ts:
                std_t = ts["std"]
                small = std_t.abs() < (MIN_NORM_RANGE / 2)
                n = int(small.sum().item())
                if n > 0:
                    ts["std"] = torch.where(
                        small,
                        torch.full_like(std_t, MIN_NORM_RANGE / 2),
                        std_t,
                    ).to(std_t.dtype)
                    log_fn(f"  [{prefix}] {feat_key}: std 提升 {n} 維 (std ≥ {MIN_NORM_RANGE/2})")

    for step in preprocessor.steps:
        _patch_step(step, "preproc")
    for step in postprocessor.steps:
        _patch_step(step, "postproc")


# ── 節點 ────────────────────────────────────────────────────────────────────

class SmolVLARTCBridgeNode(Node):
    def __init__(self, save_snapshot: bool = False, log_csv: bool = False,
                 record_video: bool = False, ckpt_path: str = FINETUNED_PATH,
                 enable_ema: bool = False, enable_home_return: bool = False,
                 ema_alpha: float = EMA_ALPHA,
                 enable_outlier_clip: bool = True,  # 預設 ON：實測 clip=0.15 解 outlier + joint7 OOD
                 clip_threshold: float = CLIP_THRESHOLD,
                 enable_gripper_snap: bool = True,
                 gripper_snap_threshold: float = GRIPPER_SNAP_THRESHOLD,
                 task: str = TASK_DESC,
                 execution_horizon: int = RTC_EXECUTION_HORIZON,
                 return_traj_npz: str = "", place_fruit: str = "", place_basket: str = ""):
        super().__init__("smolvla_rtc_bridge")
        self.task = task
        self.execution_horizon = execution_horizon
        self.save_snapshot = save_snapshot
        self.log_csv = log_csv
        self.record_video = record_video
        self.enable_ema = enable_ema
        self.enable_home_return = enable_home_return
        # D 方案 scripted return：提供 return_traj_npz 就啟用（取代 naive ramp）
        self.place_fruit = place_fruit
        self.place_basket = place_basket
        self._return_traj = None
        if return_traj_npz:
            self._return_traj = np.load(return_traj_npz)["joints"].astype(np.float64)  # (S,16)
        self._scripted_return = self._return_traj is not None
        self._return_idx = 0
        self._return_start = None
        self._place_hist: list[bool] = []
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
        log.info(f"Loading SmolVLA from {ckpt_path}...")
        self.policy = load_policy(ckpt_path, DEVICE, DTYPE)

        # 啟用 RTC：先設 config 再 init_rtc_processor（同步 policy.model.rtc_processor）
        self.policy.config.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=self.execution_horizon,
            max_guidance_weight=RTC_MAX_GUIDANCE_WEIGHT,
            prefix_attention_schedule=RTC_SCHEDULE,
        )
        self.policy.init_rtc_processor()
        log.info(
            f"RTC enabled: execution_horizon={self.execution_horizon}, "
            f"max_guidance_weight={RTC_MAX_GUIDANCE_WEIGHT}, "
            f"schedule={RTC_SCHEDULE.value}, inference_delay (init)={RTC_INFERENCE_DELAY_INIT}"
        )

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=ckpt_path,
        )
        log.info("Patching normalizer zero-variance dims (mean/std + q01/q99):")
        patch_normalizer_zero_variance(self.preprocessor, self.postprocessor, log_fn=log.info)

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

        # B 方案：publish 端 EMA filter state（16D）
        self._smoothed_action: np.ndarray | None = None  # 第一次 publish 時初始化

        # B' 方案：outlier clip — 記住前一次「實際 publish」的 action（被 clip 後的）
        self._clipped_prev: np.ndarray | None = None
        self._clip_event_count = 0  # 統計觸發次數

        # C 方案：task-done detection + home ramp state
        self._publish_frame_count = 0        # 從 settle 結束開始計
        self._grip_hist: list[float] = []    # 滑動視窗左 gripper state，長度 GRIP_WIN
        self._task_done = False
        self._home_ramp_remaining = 0
        self._home_ramp_start: np.ndarray | None = None   # ramp 起點 (16D，從當前 state 抓)

        # Video writer lock
        self._video_lock = threading.Lock()

        # Multi-episode benchmark：reset 同步
        self._episode_lock = threading.Lock()
        self._current_episode_id: int | None = None  # 從 /tmp/openarm_episode.json 讀
        self._episode_generation = 0  # 每次 reset +1，inference thread 用來辨識 stale chunk
        self._last_merge_index = None  # 上次 merge 後的 action_index，用來 gate 推論頻率（解振盪）

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

        log.info("SmolVLA RTC bridge ready (async inference + RTC inpainting)")

    # ── Multi-episode benchmark：reset 偵測 ─────────────────────────────────

    def _check_episode_file(self):
        """Poll EPISODE_FILE，發現 episode id 變化或進入 resetting 就觸發 reset。

        接受 "resetting" 跟 "ready" 兩種狀態：
          - "resetting"：Isaac 正在物理 settle，bridge 提早 settle + publish HOME，避免
            stale RTC chunk 蓋掉 Isaac 的 reset 指令
          - "ready"：Isaac reset 完成（保底觸發，防錯過 resetting）
        """
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
            return  # 沒變化（同一 episode 的 resetting → ready 不再 reset）
        self._do_episode_reset(ep_id)

    def _do_episode_reset(self, new_episode_id: int):
        """清空 policy state，為新 episode 做準備。Atomic via _episode_lock。"""
        log = self.get_logger()
        with self._episode_lock:
            log.info(f"[EPISODE] Reset triggered → episode {new_episode_id}")
            self._current_episode_id = new_episode_id
            self._episode_generation += 1
            # 重建 ActionQueue（保險，比 clear() 更乾淨）
            self.action_queue = ActionQueue(self.policy.config.rtc_config)
            self._last_merge_index = None
            # 重設 publish 端 state
            self._settle_remaining = SETTLE_FRAMES
            self._smoothed_action = None
            self._clipped_prev = None
            self._clip_event_count = 0
            self._task_done = False
            self._home_ramp_remaining = 0
            self._home_ramp_start = None
            self._grip_hist = []
            self._return_idx = 0
            self._return_start = None
            self._place_hist = []
            self._publish_frame_count = 0
            self._debug_dumped = False  # 每 episode 都 dump 一次第一幀
            self._latency_ema_frames = float(RTC_INFERENCE_DELAY_INIT)
            # 關掉 warmup_warned，下個 episode 又重新印
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
        for name, val in zip(MODEL_JOINT_NAMES_16D, state.tolist()):
            log.info(f"  {name:32s} = {val:+.4f}")
        log.info(f"[DEBUG] First chunk[0,16] = {action_np.round(3).tolist()}")
        log.info("=" * 60)

        if self.save_snapshot:
            SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            for cam_key in CAM_TOPIC_TO_KEY.values():
                img_t = raw_batch[f"observation.images.{cam_key}"]
                img_np = (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                PILImage.fromarray(img_np).save(SNAPSHOT_DIR / f"{cam_key}.png")
            with open(SNAPSHOT_DIR / "observation.json", "w") as f:
                json.dump({
                    "joint_state": dict(zip(MODEL_JOINT_NAMES_16D, state.tolist())),
                    "first_chunk_step0": dict(zip(MODEL_JOINT_NAMES_16D, action_np.tolist())),
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
            header = ["t"] + [f"state_{n}" for n in MODEL_JOINT_NAMES_16D] \
                            + [f"action_{n}" for n in MODEL_JOINT_NAMES_16D]
            self._csv_file.write(",".join(header) + "\n")
            self.get_logger().info(f"[DEBUG] CSV log started: {CSV_LOG_PATH}")
        t = self.get_clock().now().nanoseconds / 1e9
        row = [f"{t:.3f}"] + [f"{v:.6f}" for v in state_tensor.tolist()] \
                            + [f"{v:.6f}" for v in action_np]
        self._csv_file.write(",".join(row) + "\n")
        self._csv_file.flush()

    # ── 16D → 18D（mimic finger_joint2 = finger_joint1）──────────────────────

    def _action16_to_ros18(self, action16: np.ndarray) -> list[float]:
        name_to_val = dict(zip(MODEL_JOINT_NAMES_16D, action16.tolist()))
        name_to_val["openarm_left_finger_joint2"]  = name_to_val["openarm_left_finger_joint1"]
        name_to_val["openarm_right_finger_joint2"] = name_to_val["openarm_right_finger_joint1"]
        return [name_to_val[n] for n in ROS_JOINT_NAMES_18D]

    # ── Async 推論 thread（持續產 chunk + RTC merge）──────────────────────────

    def _inference_loop(self):
        log = self.get_logger()
        log.info("[INF] Inference thread started, waiting for settle to complete.")
        cam_keys = set(CAM_TOPIC_TO_KEY.values())

        # 必須等 settle 結束才開始 inference（同 PI0.5 RTC bridge 的教訓）
        while self._inference_running and self._settle_remaining > 0:
            time.sleep(0.05)
        log.info("[INF] Settle complete, starting RTC inference.")

        while self._inference_running:
            if not cam_keys.issubset(self.latest_images):
                time.sleep(0.05); continue
            if not all(n in self.latest_joint_pos for n in MODEL_JOINT_NAMES_16D):
                time.sleep(0.05); continue
            # Settle 期間（含 reset 後）暫停推論，避免推完 push 到舊 queue
            if self._settle_remaining > 0:
                time.sleep(0.05); continue

            # Inference gate：queue 還有貨時，等自上次 merge 已消耗 ≥ execution_horizon 幀才再推論。
            # 否則連續推論 → real_delay 太小 → merge 一直把 serve 位置重設回 chunk 開頭的 home
            # lead-in → publish 在 home↔大動作間振盪、永遠走不完軌跡（實測 f45=1.66 → f90=0.02）。
            if self.action_queue.qsize() > 0 and self._last_merge_index is not None:
                if (self.action_queue.get_action_index() - self._last_merge_index) < self.execution_horizon:
                    time.sleep(0.01); continue

            # Capture episode generation 為了等下檢查 chunk 是否 stale
            my_gen = self._episode_generation

            try:
                state_raw_now = torch.tensor(
                    [self.latest_joint_pos[n] for n in MODEL_JOINT_NAMES_16D], dtype=torch.float32
                )
                batch: dict = {f"observation.images.{k}": self.latest_images[k] for k in cam_keys}
                batch["observation.state"] = state_raw_now
                batch["task"] = self.task
                processed = self.preprocessor(batch)

                prev = self.action_queue.get_left_over()
                action_idx_before = self.action_queue.get_action_index()

                inference_delay_frames = max(1, int(round(self._latency_ema_frames)))

                t0 = time.time()
                # RTC denoise_step 內部用 enable_grad，必須 no_grad（不能 inference_mode）
                with torch.no_grad(), torch.autocast(DEVICE, dtype=DTYPE):
                    chunk = self.policy.predict_action_chunk(
                        processed,
                        inference_delay=inference_delay_frames,
                        prev_chunk_left_over=prev,
                    )
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                real_delay_sec = time.time() - t0

                # 用「publish 實際消耗的 frame 數」當 real_delay，避免 wallclock 在 settle 後算錯
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

                chunk_pp_16 = chunk_pp[:, :16].float()

                if not self._debug_dumped:
                    first = chunk_pp_16[0].cpu().numpy()
                    self._dump_debug(batch, processed, first)
                    self._debug_dumped = True
                    log.info(
                        f"[INF] First chunk: wallclock={real_delay_sec*1000:.0f}ms, "
                        f"queue consumed={consumed_frames} frames, "
                        f"inference_delay used={inference_delay_frames}"
                    )

                # Reset guard：若推論期間 episode reset 過，丟棄 stale chunk
                with self._episode_lock:
                    if my_gen == self._episode_generation:
                        self.action_queue.merge(
                            chunk_raw,
                            chunk_pp_16,
                            real_delay_frames,
                            action_idx_before,
                        )
                        self._last_merge_index = self.action_queue.get_action_index()
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
        """B' 方案：限制每幀 |Δaction| ≤ clip_threshold，只壓 outlier。

        prev 是上次「實際 publish 出去」的 action（不是 raw）。這樣即使連續幾幀都
        是 outlier，每次都往 raw 方向追 threshold 步長，最終會接住，不會卡死。
        """
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
        """B 方案：EMA low-pass。全 16 維一起 smooth（gripper 也要 — 不然 arm/gripper 時序錯位）。"""
        if self._smoothed_action is None:
            self._smoothed_action = raw_action.copy()
            return raw_action
        smoothed = self.ema_alpha * raw_action + (1.0 - self.ema_alpha) * self._smoothed_action
        self._smoothed_action = smoothed
        return smoothed

    def _check_task_done(self):
        """C 方案：偵測 gripper 從 closed → opened 序列觸發任務完成。"""
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

        # 條件：視窗內既有 closed (gripper 抓過 cube) 又有 opened (放開了)
        # 且最近一半 frames 是 opened 狀態（穩定 open，不是抖動）
        win = self._grip_hist
        had_closed   = min(win) < GRIP_CLOSED_TH
        recent_open  = min(win[-GRIP_WIN // 2:]) > GRIP_OPEN_TH
        if had_closed and recent_open:
            self._task_done = True
            self._home_ramp_remaining = HOME_RAMP_FRAMES
            # Ramp 起點：當前 state 16D
            self._home_ramp_start = np.asarray(
                [self.latest_joint_pos.get(n, 0.0) for n in MODEL_JOINT_NAMES_16D],
                dtype=np.float64,
            )
            self.get_logger().info(
                f"[TASK-DONE] gripper open→closed→open detected at frame "
                f"{self._publish_frame_count}. Starting {HOME_RAMP_FRAMES}-frame home ramp."
            )

    def _check_place_done(self):
        """D 方案觸發：右夾爪放開(closed→open = 放籃 release)瞬間 → 播 demo 回程軌跡。
        ★ 不能用「水果穩定入籃」觸發——policy 壞掉的回程會在水果穩定前就把它勾飛，觸發永遠不來。
        必須在 release 當下就接管。右夾爪只在放籃 open（handoff 是左夾爪 release / 右夾爪 close），
        故『右夾爪 closed→open @>300frame』= 放籃，乾淨可靠。"""
        if not self._scripted_return or self._task_done:
            return
        if self._publish_frame_count < MIN_FRAMES_BEFORE_DONE:
            return
        grip = self.latest_joint_pos.get("openarm_right_finger_joint1")
        if grip is None:
            return
        self._place_hist.append(float(grip))
        if len(self._place_hist) > GRIP_WIN:
            self._place_hist.pop(0)
        if len(self._place_hist) < GRIP_WIN:
            return
        had_closed  = min(self._place_hist) < PLACE_GRIP_CLOSED_TH    # 曾夾住(握持~0.01)
        recent_open = min(self._place_hist[-3:]) > PLACE_GRIP_OPEN_TH  # 最近 3 幀 open(~0.04)=剛 release
        if self._publish_frame_count % 60 == 0:                       # debug：看夾爪值為何沒觸發
            self.get_logger().info(
                f"[place-chk] f={self._publish_frame_count} grip={grip:.3f} "
                f"histmin={min(self._place_hist):.3f} histmax={max(self._place_hist):.3f} "
                f"had_closed={had_closed} recent_open={recent_open}")
        if had_closed and recent_open:
            self._task_done = True
            self._return_idx = 0
            self._return_start = np.asarray(
                [self.latest_joint_pos.get(n, 0.0) for n in MODEL_JOINT_NAMES_16D],
                dtype=np.float64,
            )
            self.get_logger().info(
                f"[PLACE-DONE] 右夾爪 release @frame {self._publish_frame_count}. "
                f"播 {len(self._return_traj)}-frame demo 回程軌跡。"
            )

    def _scripted_return_action(self) -> np.ndarray:
        """播 demo 回程軌跡(16D)；前 RETURN_BLEND_FRAMES 幀用 offset-decay 從當前姿態平滑接上。"""
        traj = self._return_traj
        k = self._return_idx
        if k < len(traj):
            base = traj[k]
            w = max(0.0, 1.0 - k / RETURN_BLEND_FRAMES)
            offset = (self._return_start - traj[0]) * w if self._return_start is not None else 0.0
            self._return_idx += 1
            return base + offset
        return traj[-1]  # 播完 hold home

    def _home_ramp_action(self) -> np.ndarray:
        """從 _home_ramp_start 線性 ramp 到 HOME_POSE_16D，剩餘 _home_ramp_remaining 幀。"""
        total = HOME_RAMP_FRAMES
        elapsed = total - self._home_ramp_remaining + 1
        alpha = min(1.0, elapsed / total)
        start = self._home_ramp_start if self._home_ramp_start is not None else np.zeros(16)
        target = np.asarray(HOME_POSE_16D, dtype=np.float64)
        self._home_ramp_remaining = max(0, self._home_ramp_remaining - 1)
        return (1.0 - alpha) * start + alpha * target

    # ── Publish timer（30Hz）─────────────────────────────────────────────────

    def _publish_cb(self):
        # 每幀檢查 episode reset signal（cheap：只讀一個小 JSON file）
        self._check_episode_file()

        # ── Settle phase ──
        # 永遠 publish HOME pose（不是 latest_joint_pos）。原因：
        #   reset 期間 Isaac 用 set_joint_position_target(home) 但 OmniGraph
        #   ArticulationCtrl 持續 forward SubscribeJointState 的舊指令。
        #   Bridge 在 settle 發 HOME 才能跟 Isaac reset 同步、不打架。
        if self._settle_remaining > 0:
            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name         = ROS_JOINT_NAMES_18D
            cmd.position     = self._action16_to_ros18(np.asarray(HOME_POSE_16D, dtype=np.float32))
            self.cmd_pub.publish(cmd)
            self._settle_remaining -= 1
            if self._settle_remaining == 0:
                self.get_logger().info(
                    f"Settle complete. Action queue size: {self.action_queue.qsize()} (warmed during settle)."
                )
            return

        # D 方案：偵測放籃完成（scripted return 用），每幀檢查
        self._check_place_done()

        # ── 任務完成後 → 回程（bypass policy）── scripted return 優先，否則舊 naive ramp ──
        if self._task_done:
            if self._scripted_return:
                action_np = self._scripted_return_action().astype(np.float32)
            elif self._home_ramp_remaining > 0:
                action_np = self._home_ramp_action().astype(np.float32)
            else:
                # Ramp 完成，繼續 hold home
                action_np = np.asarray(HOME_POSE_16D, dtype=np.float32)

            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name         = ROS_JOINT_NAMES_18D
            cmd.position     = self._action16_to_ros18(action_np)
            self.cmd_pub.publish(cmd)
            self._publish_frame_count += 1

            if self.log_csv and all(n in self.latest_joint_pos for n in MODEL_JOINT_NAMES_16D):
                state_now = torch.tensor(
                    [self.latest_joint_pos[n] for n in MODEL_JOINT_NAMES_16D], dtype=torch.float32
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
        if action_np.shape[0] != 16:
            action_np = action_np[:16]

        # B' 方案：outlier clip（opt-in）— 先做，壓掉異常大跳
        if self.enable_outlier_clip:
            action_np = self._apply_outlier_clip(action_np)

        # Gripper snap（opt-in，預設 ON）— policy 學到 ~0.009 半閉合，cube 會滑出，
        # 偵測「intent to close」(< snap_threshold) 直接 snap 到 0.000 全閉合。
        if self.enable_gripper_snap:
            for idx in (GRIPPER_RIGHT_IDX, GRIPPER_LEFT_IDX):
                if action_np[idx] < self.gripper_snap_threshold:
                    if action_np[idx] > GRIPPER_FULL_CLOSE:
                        self._gripper_snap_count += 1
                    action_np[idx] = GRIPPER_FULL_CLOSE

        # B 方案：EMA filter（opt-in）
        if self.enable_ema:
            action_np = self._apply_ema(action_np)

        # C 方案：task-done detection（用 raw state，不是 filtered action）
        self._publish_frame_count += 1
        self._check_task_done()

        # [DIAG] 每 15 幀印 publish 的 arm 幅度 + queue size，定位卡 home 原因
        if self._publish_frame_count % 15 == 0:
            _arm = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
            self.get_logger().info(
                f"[PUB] f={self._publish_frame_count} arm|max|={np.abs(action_np[_arm]).max():.3f} "
                f"qsize={self.action_queue.qsize()} idx={self.action_queue.get_action_index()}"
            )

        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name         = ROS_JOINT_NAMES_18D
        cmd.position     = self._action16_to_ros18(action_np)
        self.cmd_pub.publish(cmd)

        if self.log_csv and all(n in self.latest_joint_pos for n in MODEL_JOINT_NAMES_16D):
            state_now = torch.tensor(
                [self.latest_joint_pos[n] for n in MODEL_JOINT_NAMES_16D], dtype=torch.float32
            )
            self._log_frame_csv(state_now, action_np)
        if self.record_video:
            self._record_frame(self.latest_images)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_snapshot", action="store_true",
                        help="第一次推論時把影像 + state + chunk[0] 存到 /tmp/smolvla_rtc_debug/")
    parser.add_argument("--log_csv", action="store_true",
                        help="每幀 state + action 寫到 /tmp/smolvla_rtc_debug/inference_log.csv")
    parser.add_argument("--record_video", action="store_true",
                        help="每幀錄三個 camera 到 /tmp/smolvla_rtc_debug/{main,wrist_left,wrist_right}.mp4")
    parser.add_argument("--ckpt", type=str, default=FINETUNED_PATH,
                        help=f"checkpoint 路徑（預設 {FINETUNED_PATH}）")
    parser.add_argument("--enable_ema", action="store_true",
                        help="開啟 publish 端 EMA filter（降 jitter，但會延遲 1-2 frame）")
    parser.add_argument("--ema_alpha", type=float, default=EMA_ALPHA,
                        help=f"EMA 強度（0=最大平滑，1=不過濾；default={EMA_ALPHA}）")
    parser.add_argument("--disable_outlier_clip", action="store_true",
                        help="關掉 outlier clip（預設 ON：限制 |Δaction| ≤ clip_threshold 解 outlier 大跳 + joint7 OOD）")
    parser.add_argument("--clip_threshold", type=float, default=CLIP_THRESHOLD,
                        help=f"outlier clip 閾值（rad/frame；default={CLIP_THRESHOLD}）")
    parser.add_argument("--enable_home_return", action="store_true",
                        help="開啟任務完成偵測 + bypass policy 用 ramp 回 home（解 joint7 OOD attractor）")
    parser.add_argument("--disable_gripper_snap", action="store_true",
                        help=f"關掉 gripper snap（預設 ON：policy gripper output < {GRIPPER_SNAP_THRESHOLD} 時 snap 到 0.0 全閉合，解 teleop 沒夾緊問題）")
    parser.add_argument("--gripper_snap_threshold", type=float, default=GRIPPER_SNAP_THRESHOLD,
                        help=f"Gripper snap 閾值（default={GRIPPER_SNAP_THRESHOLD}）")
    parser.add_argument("--task", type=str, default=TASK_DESC,
                        help=f"task 語言字串（★ 必須跟 dataset tasks.parquet 完全一致；多任務 handover 用此切換）。default={TASK_DESC!r}")
    parser.add_argument("--execution_horizon", type=int, default=RTC_EXECUTION_HORIZON,
                        help=f"RTC 每次 commit 幾幀後重推論。trimmed dataset 開頭有 ~10 幀 lead-in(near-home)，"
                             f"horizon<=10 會卡在 home 永遠到不了動作段 → 需調大(建議 25~30)。default={RTC_EXECUTION_HORIZON}")
    parser.add_argument("--return_traj_npz", type=str, default="",
                        help="D 方案 scripted return：demo 回程軌跡 npz（有 'joints' (S,16)）。"
                             "提供就啟用——偵測水果入籃後 bypass policy 播此軌跡回 home（取代會撞籃子的 policy 回程）")
    parser.add_argument("--place_fruit", type=str, default="",
                        help="scripted return 觸發用：目標水果名（apple/strawberry/banana）")
    parser.add_argument("--place_basket", type=str, default="",
                        help="scripted return 觸發用：目標籃名（red/yellow）")
    args, ros_argv = parser.parse_known_args()

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
        task=args.task,
        execution_horizon=args.execution_horizon,
        return_traj_npz=args.return_traj_npz,
        place_fruit=args.place_fruit,
        place_basket=args.place_basket,
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
