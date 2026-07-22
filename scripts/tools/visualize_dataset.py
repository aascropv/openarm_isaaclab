#!/usr/bin/env python3
"""
OpenArm Dataset Visualizer
顯示 collect_demonstrations.py 收集的 HDF5 示範資料。

使用方式：
    python3 scripts/visualize_dataset.py [--dataset_dir ./datasets]
"""

import argparse
import os
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageTk

# ─── 關節名稱（與 collect_demonstrations.py 的關節順序對應） ───────────────
# Isaac Articulation 的 joint 順序由 USD 決定，以下根據實際 18 個 joint 排列
JOINT_NAMES = (
    [f"L_arm{i}" for i in range(1, 8)]    # openarm_left_joint1-7
    + [f"L_grip{i}" for i in range(1, 3)] # openarm_left_finger_joint1-2
    + [f"R_arm{i}" for i in range(1, 8)]  # openarm_right_joint1-7
    + [f"R_grip{i}" for i in range(1, 3)] # openarm_right_finger_joint1-2
)

# 各區段的 index 範圍（用於在表格加分隔線）
_SECTIONS = [
    ("Left Arm",    range(0, 7)),
    ("Left Gripper",range(7, 9)),
    ("Right Arm",   range(9, 16)),
    ("Right Gripper",range(16, 18)),
]


class DatasetViewer:
    def __init__(self, root: tk.Tk, dataset_dir: str):
        self.root = root
        self.root.title("OpenArm Dataset Viewer")
        self.dataset_dir = Path(dataset_dir)

        # 目前載入的資料
        self.hdf5: h5py.File | None = None
        self.num_frames = 0
        self.frame_idx = 0

        self._build_ui()
        self._refresh_run_list()

    # ─── UI 建構 ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # ── 左側面板：選擇器 + 資訊 ──
        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="ns")

        ttk.Label(left, text="Run").pack(anchor="w")
        self.run_var = tk.StringVar()
        self.run_cb = ttk.Combobox(left, textvariable=self.run_var, width=22, state="readonly")
        self.run_cb.pack(fill="x")
        self.run_cb.bind("<<ComboboxSelected>>", lambda _: self._on_run_changed())

        ttk.Label(left, text="Episode").pack(anchor="w", pady=(8, 0))
        self.ep_var = tk.StringVar()
        self.ep_cb = ttk.Combobox(left, textvariable=self.ep_var, width=22, state="readonly")
        self.ep_cb.pack(fill="x")
        self.ep_cb.bind("<<ComboboxSelected>>", lambda _: self._on_episode_changed())

        # Episode 資訊
        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)
        self.info_var = tk.StringVar(value="—")
        ttk.Label(left, textvariable=self.info_var, justify="left", wraplength=170).pack(anchor="w")

        # 播放控制
        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)
        ctrl = ttk.Frame(left)
        ctrl.pack(fill="x")
        self.play_btn = ttk.Button(ctrl, text="▶ Play", command=self._toggle_play)
        self.play_btn.pack(side="left")
        ttk.Label(ctrl, text="FPS").pack(side="left", padx=(8, 2))
        self.fps_var = tk.IntVar(value=30)
        ttk.Spinbox(ctrl, from_=1, to=60, textvariable=self.fps_var, width=4).pack(side="left")

        self.playing = False
        self._play_job = None

        # ── 右側主區域 ──
        main = ttk.Frame(self.root, padding=4)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)
        main.rowconfigure(1, weight=0)
        main.rowconfigure(2, weight=1)

        # 相機畫面區
        cam_frame = ttk.LabelFrame(main, text="Cameras")
        cam_frame.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        for i in range(3):
            cam_frame.columnconfigure(i, weight=1)

        cam_labels_info = [("Main", 0), ("Wrist Left", 1), ("Wrist Right", 2)]
        self.cam_labels: list[ttk.Label] = []
        self.cam_tk: list[ImageTk.PhotoImage | None] = [None, None, None]
        for title, col in cam_labels_info:
            f = ttk.Frame(cam_frame)
            f.grid(row=0, column=col, padx=4, pady=4)
            ttk.Label(f, text=title).pack()
            lbl = ttk.Label(f)
            lbl.pack()
            self.cam_labels.append(lbl)

        # 進度條 + 幀號
        slider_frame = ttk.Frame(main)
        slider_frame.grid(row=1, column=0, sticky="ew", pady=4)
        slider_frame.columnconfigure(0, weight=1)

        self.frame_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(
            slider_frame, from_=0, to=0, orient="horizontal",
            variable=self.frame_var, command=self._on_slider
        )
        self.slider.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.frame_label = ttk.Label(slider_frame, text="0 / 0")
        self.frame_label.grid(row=0, column=1)

        # 關節 + Cube 數值顯示（Label grid，無捲軸）
        data_frame = ttk.LabelFrame(main, text="Joint Positions & Cube")
        data_frame.grid(row=2, column=0, sticky="nsew", pady=(4, 0))

        FONT = ("Courier", 9)
        HDR  = ("Courier", 9, "bold")
        COL_HEADERS = ("Name", "Pos(rad)", "Vel(r/s)")

        # 每個 section 放在獨立的子 frame，左右兩欄排列
        # 左欄：Left Arm + Left Gripper；右欄：Right Arm + Right Gripper；右側：Cube
        left_col  = ttk.Frame(data_frame)
        right_col = ttk.Frame(data_frame)
        cube_col  = ttk.Frame(data_frame)
        left_col.grid( row=0, column=0, sticky="n", padx=8, pady=4)
        right_col.grid(row=0, column=1, sticky="n", padx=8, pady=4)
        cube_col.grid( row=0, column=2, sticky="n", padx=8, pady=4)

        # val_labels[i] = (pos_lbl, vel_lbl) for joint i
        self._val_labels: list[tuple[tk.Label, tk.Label]] = []

        def _make_section(parent, title, joint_indices, start_row):
            """在 parent 從 start_row 開始建立一個區段，回傳下一個可用 row。"""
            # 區段標題
            tk.Label(parent, text=title, font=HDR, anchor="w").grid(
                row=start_row, column=0, columnspan=4, sticky="w", pady=(4, 0))
            r = start_row + 1
            # 欄位標頭
            for c, h in enumerate(COL_HEADERS):
                tk.Label(parent, text=h, font=HDR, padx=4).grid(row=r, column=c, sticky="e")
            r += 1
            # 資料列
            for i in joint_indices:
                tk.Label(parent, text=JOINT_NAMES[i], font=FONT, padx=4, anchor="w").grid(
                    row=r, column=0, sticky="w")
                pos_lbl = tk.Label(parent, text="—", font=FONT, padx=4, anchor="e", width=10)
                vel_lbl = tk.Label(parent, text="—", font=FONT, padx=4, anchor="e", width=10)
                pos_lbl.grid(row=r, column=1, sticky="e")
                vel_lbl.grid(row=r, column=2, sticky="e")
                self._val_labels.append((pos_lbl, vel_lbl))
                r += 1
            return r

        row_l = 0
        row_l = _make_section(left_col,  "Left Arm",     range(0, 7),  row_l)
        row_l = _make_section(left_col,  "Left Gripper", range(7, 9),  row_l)

        row_r = 0
        row_r = _make_section(right_col, "Right Arm",     range(9, 16), row_r)
        row_r = _make_section(right_col, "Right Gripper", range(16, 18),row_r)

        # Cube 數值 Labels
        tk.Label(cube_col, text="Cube Position", font=HDR, anchor="w").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self._cube_labels: dict[str, tk.Label] = {}
        cube_rows = [("X (m)", "cx"), ("Y (m)", "cy"), ("Z (m)", "cz"),
                     ("", None),
                     ("Quaternion", None),
                     ("qx", "qx"), ("qy", "qy"), ("qz", "qz"), ("qw", "qw")]
        r = 1
        for label_text, key in cube_rows:
            if key is None:
                if label_text:  # 小標題
                    tk.Label(cube_col, text=label_text, font=HDR, anchor="w").grid(
                        row=r, column=0, columnspan=2, sticky="w", pady=(4, 0))
                r += 1
                continue
            tk.Label(cube_col, text=label_text, font=FONT, padx=4, anchor="w").grid(
                row=r, column=0, sticky="w")
            val_lbl = tk.Label(cube_col, text="—", font=FONT, padx=4, anchor="e", width=12)
            val_lbl.grid(row=r, column=1, sticky="e")
            self._cube_labels[key] = val_lbl
            r += 1

    # ─── 資料夾 / Episode 選擇 ────────────────────────────────────────────────

    def _refresh_run_list(self):
        runs = sorted(
            [d.name for d in self.dataset_dir.iterdir() if d.is_dir()],
            reverse=True,
        )
        self.run_cb["values"] = runs
        if runs:
            self.run_var.set(runs[0])
            self._on_run_changed()

    def _on_run_changed(self):
        run_name = self.run_var.get()
        run_path = self.dataset_dir / run_name
        eps = sorted(run_path.glob("demo_*.hdf5"))
        ep_names = [e.name for e in eps]
        self.ep_cb["values"] = ep_names
        if ep_names:
            self.ep_var.set(ep_names[0])
            self._on_episode_changed()

    def _on_episode_changed(self):
        self._stop_play()
        run_name = self.run_var.get()
        ep_name = self.ep_var.get()
        path = self.dataset_dir / run_name / ep_name

        if self.hdf5 is not None:
            self.hdf5.close()
            self.hdf5 = None

        self.hdf5 = h5py.File(path, "r")
        self.num_frames = int(self.hdf5.attrs.get("num_steps", 0))

        # 更新 episode 資訊
        success = bool(self.hdf5.attrs.get("success", False))
        cube_xyz = self.hdf5.attrs.get("cube_xyz", [0, 0, 0])
        self.info_var.set(
            f"Frames : {self.num_frames}\n"
            f"Success: {'✓' if success else '✗'}\n"
            f"Cube   : ({cube_xyz[0]:.3f}, {cube_xyz[1]:.3f}, {cube_xyz[2]:.3f})"
        )

        # 更新 slider 範圍
        self.slider.config(to=max(0, self.num_frames - 1))
        self.frame_var.set(0)
        self.frame_idx = 0

        # 預先載入全部關節/cube 資料（小資料，不佔太多記憶體）
        self.all_joint_pos  = self.hdf5["obs/joint_pos"][:]   # (N, 18)
        self.all_joint_vel  = self.hdf5["obs/joint_vel"][:]   # (N, 18)
        self.all_cube_pos   = self.hdf5["obs/cube_pos"][:]    # (N, 3)
        self.all_cube_quat  = self.hdf5["obs/cube_quat"][:]   # (N, 4)


        self._render_frame(0)

    # ─── 幀渲染 ──────────────────────────────────────────────────────────────

    def _on_slider(self, val):
        idx = int(float(val))
        if idx != self.frame_idx:
            self.frame_idx = idx
            self._render_frame(idx)

    def _render_frame(self, idx: int):
        if self.hdf5 is None or self.num_frames == 0:
            return
        idx = max(0, min(idx, self.num_frames - 1))

        # ── 相機影像 ──
        cam_keys = ["obs/cam_main", "obs/cam_wrist_left", "obs/cam_wrist_right"]
        for i, key in enumerate(cam_keys):
            raw = self.hdf5[key][idx]           # (H, W, 3)
            img = Image.fromarray(raw.astype(np.uint8))
            img = img.resize((320, 240), Image.NEAREST)
            photo = ImageTk.PhotoImage(img)
            self.cam_labels[i].config(image=photo)
            self.cam_tk[i] = photo              # 防止 GC 回收

        # ── 幀號標籤 ──
        self.frame_label.config(text=f"{idx + 1} / {self.num_frames}")

        # ── 數值文字 ──
        self._update_data_text(idx)

    def _update_data_text(self, idx: int):
        jpos = self.all_joint_pos[idx]
        jvel = self.all_joint_vel[idx]
        for i, (pos_lbl, vel_lbl) in enumerate(self._val_labels):
            pos_lbl.config(text=f"{jpos[i]:.4f}")
            vel_lbl.config(text=f"{jvel[i]:.4f}")

        cp = self.all_cube_pos[idx]
        cq = self.all_cube_quat[idx]
        self._cube_labels["cx"].config(text=f"{cp[0]:.5f}")
        self._cube_labels["cy"].config(text=f"{cp[1]:.5f}")
        self._cube_labels["cz"].config(text=f"{cp[2]:.5f}")
        self._cube_labels["qx"].config(text=f"{cq[0]:.5f}")
        self._cube_labels["qy"].config(text=f"{cq[1]:.5f}")
        self._cube_labels["qz"].config(text=f"{cq[2]:.5f}")
        self._cube_labels["qw"].config(text=f"{cq[3]:.5f}")

    # ─── 播放控制 ─────────────────────────────────────────────────────────────

    def _toggle_play(self):
        if self.playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        self.playing = True
        self.play_btn.config(text="⏸ Pause")
        self._play_step()

    def _stop_play(self):
        self.playing = False
        self.play_btn.config(text="▶ Play")
        if self._play_job is not None:
            self.root.after_cancel(self._play_job)
            self._play_job = None

    def _play_step(self):
        if not self.playing:
            return
        next_idx = self.frame_idx + 1
        if next_idx >= self.num_frames:
            self._stop_play()
            return
        self.frame_idx = next_idx
        self.frame_var.set(next_idx)
        self._render_frame(next_idx)
        delay_ms = max(1, 1000 // self.fps_var.get())
        self._play_job = self.root.after(delay_ms, self._play_step)

    # ─── 清理 ────────────────────────────────────────────────────────────────

    def on_close(self):
        self._stop_play()
        if self.hdf5 is not None:
            self.hdf5.close()
        self.root.destroy()


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OpenArm Dataset Viewer")
    parser.add_argument("--dataset_dir", default="./datasets",
                        help="資料集根目錄（預設 ./datasets）")
    args = parser.parse_args()

    root = tk.Tk()
    root.geometry("1280x780")
    app = DatasetViewer(root, args.dataset_dir)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
