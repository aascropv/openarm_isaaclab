#!/usr/bin/env python3
"""dataset_viewer.py — 逐幀檢視「訓練示範資料」(HDF5 demo)，介面內可直接切 episode。

顯示：三相機並排(main|wrist_left|wrist_right) + 16 關節 bar + 水果 xyz + success。
資料來源：datasets_handover_v2/<fruit>/<fruit>_<timestamp>/demo_*.hdf5（每檔一集，含 obs/cam_* + joint_pos + cube_pos）。

★ 效能：HDF5 相機是 gzip+chunk，隨機讀單幀極慢 → 切集時一次把整集讀進 RAM，播放才順。
★ 中文：用系統 Noto Sans CJK 字型，避免 tofu 方框。
★ 重畫：影像/bar 只建一次，用 set_data/set_width 更新（不每幀 clear+imshow）。

用法:
  conda run -n lerobot python3 scripts/tools/dataset_viewer.py --fruit apple
  conda run -n lerobot python3 scripts/tools/dataset_viewer.py --dir datasets_handover_v2/apple --episode 5
"""
import argparse
import glob
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox

# ── 中文字型（避免方框）──
for _name in ["Noto Sans CJK JP", "Noto Sans CJK TC", "Noto Sans CJK SC", "WenQuanYi Zen Hei"]:
    if any(_name == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_name, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

JOINT_LABELS = ([f"R_j{i}" for i in range(1, 8)] + ["R_grip"] +
                [f"L_j{i}" for i in range(1, 8)] + ["L_grip"])
CAMS = ["cam_main", "cam_wrist_left", "cam_wrist_right"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruit", default=None, help="apple/strawberry/banana → datasets_handover_v2/<fruit>")
    ap.add_argument("--dir", default=None, help="直接給 demo_*.hdf5 所在目錄（覆蓋 --fruit）")
    ap.add_argument("--episode", type=int, default=0, help="起始 episode index")
    a = ap.parse_args()

    ddir = a.dir or (f"datasets_handover_v2/{a.fruit}" if a.fruit else None)
    if not ddir:
        ap.error("需給 --fruit 或 --dir")
    files = sorted(glob.glob(str(Path(ddir) / "demo_*.hdf5")))
    if not files:   # v2 佈局多一層時間戳：<fruit>/<fruit>_<timestamp>/demo_*.hdf5
        files = sorted(glob.glob(str(Path(ddir) / "*" / "demo_*.hdf5")))
    if not files:
        ap.error(f"找不到 demo_*.hdf5 於 {ddir}（也試過 {ddir}/*/）")
    print(f"{len(files)} episodes 在 {ddir}")

    # 整集資料快取（RAM）
    S = {"ep": -1, "combo": None, "jp": None, "cube": None, "n": 0, "attrs": {}}

    def load_ep(ei):
        """一次把整集 3 相機循序讀進 RAM（快），並排成 [n,240,960,3]。"""
        ei = max(0, min(ei, len(files) - 1))
        with h5py.File(files[ei], "r") as f:
            cams = [f[f"obs/{c}"][:] for c in CAMS]        # 循序讀，每 chunk 只解壓一次
            S["jp"] = f["obs/joint_pos"][:]
            S["cube"] = f["obs/cube_pos"][:]
            S["attrs"] = {k: f.attrs[k] for k in f.attrs}
        S["combo"] = np.concatenate(cams, axis=2)          # 水平並排
        S["n"] = S["combo"].shape[0]
        S["ep"] = ei

    # ── figure ──
    fig = plt.figure(figsize=(14, 7))
    fig.canvas.manager.set_window_title(f"dataset viewer — {ddir}")
    ax_img = fig.add_axes([0.03, 0.40, 0.66, 0.55]); ax_img.axis("off")
    ax_j = fig.add_axes([0.75, 0.40, 0.22, 0.55])
    ax_sl = fig.add_axes([0.07, 0.28, 0.60, 0.03])

    # 顯示「載入中」再讀第一集（讀 1.25GB 需 1~2 秒）
    ax_img.text(0.5, 0.5, "載入中…", ha="center", va="center", fontsize=16, transform=ax_img.transAxes)
    fig.canvas.draw(); plt.pause(0.001)
    load_ep(max(0, min(a.episode, len(files) - 1)))
    ax_img.clear(); ax_img.axis("off")

    # 只建一次的 artists
    im = ax_img.imshow(S["combo"][0])
    w = S["combo"].shape[2] // 3
    for j, lab in enumerate(["overhead(main)", "wrist_left", "wrist_right"]):
        ax_img.text(w * j + w / 2, -8, lab, ha="center", fontsize=9)
    bars = ax_j.barh(range(16), S["jp"][0],
                     color=["tab:blue"] * 8 + ["tab:orange"] * 8)
    ax_j.set_yticks(range(16)); ax_j.set_yticklabels(JOINT_LABELS, fontsize=7)
    ax_j.axvline(0, c="k", lw=.5); ax_j.invert_yaxis()
    ax_j.set(xlabel="rad", title="16 關節 (R藍 L橘)", xlim=(-2.2, 2.2)); ax_j.grid(alpha=.3, axis="x")

    slider = Slider(ax_sl, "frame", 0, S["n"] - 1, valinit=0, valstep=1)
    bplay = Button(fig.add_axes([0.72, 0.275, 0.09, 0.045]), "▶ play")
    bprev = Button(fig.add_axes([0.07, 0.20, 0.10, 0.05]), "◀ ep")
    bnext = Button(fig.add_axes([0.19, 0.20, 0.10, 0.05]), "ep ▶")
    tb = TextBox(fig.add_axes([0.42, 0.20, 0.10, 0.05]), "跳到 ep ", initial=str(S["ep"]))

    def draw(i):
        i = int(i)
        im.set_data(S["combo"][i])
        for rect, v in zip(bars, S["jp"][i]):
            rect.set_width(v)
        cx, cy, cz = S["cube"][i]
        at = S["attrs"]
        fr = at.get("fruit", a.fruit or "?"); bk = at.get("basket", "?"); su = at.get("success", "?")
        fig.suptitle(f"{fr}→{bk}   ep {S['ep']}/{len(files)-1}   frame {i}/{S['n']-1}   |   "
                     f"{fr} xyz=({cx:.3f},{cy:.3f},{cz:.3f})   success={su}",
                     fontsize=12, fontweight="bold")
        fig.canvas.draw_idle()

    slider.on_changed(draw)

    def switch(ei):
        ei = max(0, min(ei, len(files) - 1))
        if ei == S["ep"]:
            return
        fig.suptitle(f"載入 ep {ei} 中…", fontsize=12); fig.canvas.draw(); plt.pause(0.001)
        load_ep(ei)
        tb.eventson = False; tb.set_val(str(ei)); tb.eventson = True
        slider.valmax = S["n"] - 1
        slider.ax.set_xlim(0, S["n"] - 1)
        if int(slider.val) == 0:
            draw(0)
        else:
            slider.set_val(0)   # 觸發 draw(0)

    bprev.on_clicked(lambda e: switch(S["ep"] - 1))
    bnext.on_clicked(lambda e: switch(S["ep"] + 1))
    def on_submit(txt):
        try:
            switch(int(txt))
        except ValueError:
            pass
    tb.on_submit(on_submit)

    st = {"play": False, "step": 3}
    timer = fig.canvas.new_timer(interval=33)
    def _tick():
        if not st["play"]:
            return
        v = int(slider.val) + st["step"]
        if v >= S["n"] - 1:
            v = S["n"] - 1; st["play"] = False; bplay.label.set_text("▶ play"); timer.stop()
        slider.set_val(v)
    timer.add_callback(_tick)
    def toggle(_e):
        st["play"] = not st["play"]
        if st["play"]:
            if int(slider.val) >= S["n"] - 1:
                slider.set_val(0)
            bplay.label.set_text("⏸ pause"); timer.start()
        else:
            bplay.label.set_text("▶ play"); timer.stop()
    bplay.on_clicked(toggle)

    draw(0)
    print("介面：frame slider 拉幀 / ▶play 自動播 / ◀ep ep▶ 換集 / 「跳到 ep」打數字 Enter 直接跳")
    plt.show()


if __name__ == "__main__":
    main()
