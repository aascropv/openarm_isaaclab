#!/usr/bin/env python3
"""eval_menu_viewer.py — 選單式 eval 檢視 GUI（水果 → ckpt → episode 下拉選）。

自動掃描 outputs/evalview_*（eval_record_view_act.sh 產生的 <base>.csv + <base>_3cam.mp4），
下拉選 水果/ckpt/episode，逐幀看：三相機並排 + 16 關節 + 雙臂側視(x-z) + 該集入籃/回home。
錄影還在陸續產生時，按 🔄 rescan 重新掃描即可看到新出爐的 ckpt。

用法:
  conda run -n lerobot python3 scripts/tools/eval_menu_viewer.py
"""
import glob
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tkinter as tk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PROJ = Path("/home/teddyminelab/openarm_isaaclab")
RB = dict(x=(0.198, 0.349), y=(-0.115, -0.031))
YB = dict(x=(0.198, 0.349), y=(-0.221, -0.136))
PLAT = dict(x=(0.166, 0.568), top=0.193)
BASK = {"red": RB, "yellow": YB}
FRUIT_BASKET = {"apple": "red", "strawberry": "red", "banana": "yellow"}


def discover():
    """掃 v2 recording → {fruit: {ckpt: base_path}}（需同名 .csv）。

    嚴格只認這次 v2 sweep 的命名 evalview_act_handover_<fruit>_v2_<ckpt>，
    避免混入舊 v1 / clean53 / camera-test 的 recording。
    """
    idx = {}
    for mp4 in sorted(glob.glob(str(PROJ / "outputs" / "evalview_act_handover_*_v2_*_3cam.mp4"))):
        base = mp4[: -len("_3cam.mp4")]
        if not os.path.exists(base + ".csv"):
            continue
        name = os.path.basename(base)
        m = re.match(r"evalview_act_handover_(apple|strawberry|banana)_v2_(\d{6})$", name)
        if not m:
            continue
        fruit, ckpt = m.group(1), m.group(2)
        idx.setdefault(fruit, {})[ckpt] = base
    return idx


def arm_links(df, arm):
    order = [f"openarm_{arm}_link{i}" for i in range(1, 8)] + \
            [f"openarm_{arm}_left_finger", f"openarm_{arm}_right_finger"]
    return [b for b in order if f"{b}_x" in df.columns]


class Viewer:
    def __init__(self):
        self.idx = discover()
        self.df = None
        self.cap = None
        self.ts_wall = None
        self.episodes = []
        self.ep_start = {}
        self.bench = {}
        self.JOINTS = self.RIGHT = self.LEFT = self.FRUITS = []
        self.fruit = self.ckpt = None
        self.basket = "red"
        self.playing = False

        self.root = tk.Tk()
        self.root.title("eval menu viewer — v2 handover")

        top = tk.Frame(self.root, pady=4)
        top.pack(side=tk.TOP, fill=tk.X)
        tk.Label(top, text="水果").pack(side=tk.LEFT, padx=(6, 2))
        self.v_fruit = tk.StringVar()
        self.m_fruit = tk.OptionMenu(top, self.v_fruit, "")
        self.m_fruit.config(width=10)
        self.m_fruit.pack(side=tk.LEFT)
        tk.Label(top, text="ckpt").pack(side=tk.LEFT, padx=(10, 2))
        self.v_ckpt = tk.StringVar()
        self.m_ckpt = tk.OptionMenu(top, self.v_ckpt, "")
        self.m_ckpt.config(width=8)
        self.m_ckpt.pack(side=tk.LEFT)
        tk.Label(top, text="episode").pack(side=tk.LEFT, padx=(10, 2))
        self.v_ep = tk.StringVar()
        self.m_ep = tk.OptionMenu(top, self.v_ep, "")
        self.m_ep.config(width=22)
        self.m_ep.pack(side=tk.LEFT)
        tk.Button(top, text="◀", command=lambda: self.step(-2)).pack(side=tk.LEFT, padx=(10, 0))
        tk.Button(top, text="▶", command=lambda: self.step(2)).pack(side=tk.LEFT)
        self.b_play = tk.Button(top, text="▶ play", command=self.toggle_play)
        self.b_play.pack(side=tk.LEFT, padx=6)
        tk.Button(top, text="🔄 rescan", command=self.rescan).pack(side=tk.RIGHT, padx=6)

        self.fig = Figure(figsize=(15, 8))
        self.ax_img = self.fig.add_axes([0.03, 0.42, 0.94, 0.52]); self.ax_img.axis("off")
        self.ax_side = self.fig.add_axes([0.06, 0.10, 0.40, 0.26])
        self.ax_j = self.fig.add_axes([0.55, 0.10, 0.40, 0.26])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.v_frame = tk.IntVar(value=0)
        self.slider = tk.Scale(self.root, from_=0, to=1, orient=tk.HORIZONTAL,
                               variable=self.v_frame, command=self.on_slider, label="frame")
        self.slider.pack(side=tk.BOTTOM, fill=tk.X)

        self._populate_fruits()

    # ── 選單填充 ───────────────────────────────────────────────────────────
    def _set_menu(self, om, var, options, command):
        menu = om["menu"]
        menu.delete(0, "end")
        for o in options:
            menu.add_command(label=o, command=lambda val=o: command(val))
        var.set(options[0] if options else "")

    def _populate_fruits(self):
        fruits = list(self.idx.keys())
        if not fruits:
            self._set_menu(self.m_fruit, self.v_fruit, ["(尚無 recording)"], lambda v: None)
            self._msg("尚無 recording。錄影還在跑就等一下，好了按 🔄 rescan。")
            return
        self._set_menu(self.m_fruit, self.v_fruit, fruits, self.on_fruit)
        self.on_fruit(fruits[0])

    def on_fruit(self, fruit):
        self.v_fruit.set(fruit)
        ckpts = sorted(self.idx.get(fruit, {}).keys())
        self._set_menu(self.m_ckpt, self.v_ckpt, ckpts, self.on_ckpt)
        if ckpts:
            self.on_ckpt(ckpts[0])

    def on_ckpt(self, ckpt):
        self.v_ckpt.set(ckpt)
        fruit = self.v_fruit.get()
        self.load(fruit, ckpt)

    def on_ep(self, label):
        self.v_ep.set(label)
        ep = int(label.split()[0][2:])   # "ep3 ..." → 3
        if ep in self.ep_start:
            self.v_frame.set(int(self.ep_start[ep]))
            self.draw(int(self.ep_start[ep]))

    def rescan(self):
        keep = (self.v_fruit.get(), self.v_ckpt.get())
        self.idx = discover()
        self._populate_fruits()
        # 盡量回到原本選的
        f, c = keep
        if f in self.idx:
            self.on_fruit(f)
            if c in self.idx.get(f, {}):
                self.on_ckpt(c)

    # ── 載入一組 recording ─────────────────────────────────────────────────
    def load(self, fruit, ckpt):
        self.fruit, self.ckpt = fruit, ckpt
        self.basket = FRUIT_BASKET[fruit]
        base = self.idx[fruit][ckpt]
        df = pd.read_csv(base + ".csv")
        df = df[df.state.isin(["ready", "running"])].reset_index(drop=True)
        self.df = df
        self.JOINTS = [c for c in df.columns
                       if c.startswith("openarm_") and "joint" in c and not c.endswith(("_x", "_y", "_z"))]
        self.RIGHT, self.LEFT = arm_links(df, "right"), arm_links(df, "left")
        self.FRUITS = [f for f in ["apple", "strawberry", "banana"] if f"{f}_x" in df.columns]
        self.episodes = sorted(df.episode.unique())
        self.ep_start = {ep: df.index[df.episode == ep][0] for ep in self.episodes}

        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(base + "_3cam.mp4")
        try:
            ts = pd.read_csv(base + "_3cam.mp4.timestamps.csv")
            self.ts_wall = ts["wall"].values
        except Exception:
            self.ts_wall = None

        # 權威 benchmark 結果（success + arm_returned_home）
        self.bench = {}
        bfiles = sorted(glob.glob(
            f"results_handover_eval/handover_{fruit}_{self.basket}_recview_act_{ckpt}_*.csv"))
        if bfiles:
            for _, br in pd.read_csv(bfiles[-1]).iterrows():
                self.bench[int(br["episode"])] = (
                    str(br["success"]).strip().lower() == "true",
                    str(br["arm_returned_home"]).strip().lower() == "true")

        # episode 下拉（附狀態）
        labels = []
        for ep in self.episodes:
            inb, home = self.ep_status(ep)
            labels.append(f"ep{int(ep)}  in={'Y' if inb else 'N'} home={'Y' if home else 'N'}")
        self._set_menu(self.m_ep, self.v_ep, labels, self.on_ep)

        self.slider.config(to=max(len(df) - 1, 1))
        self.v_frame.set(0)
        self.draw(0)

    def ep_status(self, ep):
        if int(ep) in self.bench:
            return self.bench[int(ep)]
        e = self.df[self.df.episode == ep]
        last = e.iloc[-1]
        b = BASK[self.basket]
        fx, fy, fz = last[f"{self.fruit}_x"], last[f"{self.fruit}_y"], last[f"{self.fruit}_z"]
        inb = b["x"][0] <= fx <= b["x"][1] and b["y"][0] <= fy <= b["y"][1] and fz < 0.28
        arms = [max(abs(row[j]) for j in self.JOINTS if "finger" not in j)
                for _, row in e.tail(30).iterrows()]
        return inb, (min(arms) < 0.20 if arms else False)

    # ── 影像對齊 ───────────────────────────────────────────────────────────
    def frame_img(self, row_wall):
        if self.ts_wall is None or self.cap is None or not self.cap.isOpened():
            return None
        k = int(np.argmin(np.abs(self.ts_wall - row_wall)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, im = self.cap.read()
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if ok else None

    def _msg(self, text):
        self.ax_img.clear(); self.ax_img.axis("off")
        self.ax_img.text(0.5, 0.5, text, ha="center", va="center",
                         transform=self.ax_img.transAxes, fontsize=13)
        self.canvas.draw_idle()

    # ── 繪製 ───────────────────────────────────────────────────────────────
    def draw(self, i):
        if self.df is None or len(self.df) == 0:
            return
        i = max(0, min(int(i), len(self.df) - 1))
        r = self.df.iloc[i]
        ep, fr = int(r.episode), int(r.frame)
        inb, home = self.ep_status(ep)

        img = self.frame_img(r.wall) if "wall" in self.df.columns else None
        self.ax_img.clear(); self.ax_img.axis("off")
        if img is not None:
            self.ax_img.imshow(img)
            w = img.shape[1] // 3
            for j, lab in enumerate(["overhead", "wrist_left", "wrist_right"]):
                self.ax_img.text(w * j + w / 2, -8, lab, ha="center", fontsize=10)
        else:
            self.ax_img.text(0.5, 0.5, "(無影像 / mp4 對不上)", ha="center", va="center",
                             transform=self.ax_img.transAxes)

        self.ax_side.clear()
        for links, c in [(self.RIGHT, "tab:blue"), (self.LEFT, "tab:orange")]:
            self.ax_side.plot([r[f"{b}_x"] for b in links], [r[f"{b}_z"] for b in links],
                              "-o", color=c, ms=3)
        for f in self.FRUITS:
            self.ax_side.scatter([r[f"{f}_x"]], [r[f"{f}_z"]], s=90, marker="*", zorder=5)
        self.ax_side.add_patch(Rectangle((PLAT["x"][0], 0), PLAT["x"][1] - PLAT["x"][0],
                                         PLAT["top"], color="brown", alpha=.18))
        self.ax_side.axhline(PLAT["top"], ls=":", c="brown")
        self.ax_side.set(xlabel="x", ylabel="z", title="side x-z (platform hit?)",
                         xlim=(-0.1, 0.7), ylim=(0, 0.65))
        self.ax_side.grid(alpha=.3)

        self.ax_j.clear()
        names = [j.replace("openarm_", "").replace("_joint", "_j").replace("_finger_j1", "_grip")
                 for j in self.JOINTS]
        self.ax_j.barh(range(len(self.JOINTS)), [r[j] for j in self.JOINTS],
                       color=["tab:blue" if "right" in j else "tab:orange" for j in self.JOINTS])
        self.ax_j.set_yticks(range(len(self.JOINTS)))
        self.ax_j.set_yticklabels(names, fontsize=6)
        self.ax_j.axvline(0, c="k", lw=.5); self.ax_j.invert_yaxis()
        self.ax_j.set(xlabel="rad", title="16 joints", xlim=(-2.2, 2.2))
        self.ax_j.grid(alpha=.3, axis="x")

        self.fig.suptitle(
            f"{self.fruit} {self.ckpt}   ep{ep}  frame {fr}   |   "
            f"in-basket={'YES' if inb else 'NO'}   arm-home={'YES' if home else 'NO'}   "
            f"(blue=right, orange=left)", fontsize=13, fontweight="bold")
        self.canvas.draw_idle()

    # ── 互動 ───────────────────────────────────────────────────────────────
    def on_slider(self, val):
        if not self.playing:
            self.draw(int(float(val)))

    def step(self, d):
        self.v_frame.set(max(0, min(int(self.v_frame.get()) + d,
                                    (len(self.df) - 1) if self.df is not None else 0)))
        self.draw(int(self.v_frame.get()))

    def toggle_play(self):
        self.playing = not self.playing
        self.b_play.config(text="⏸ pause" if self.playing else "▶ play")
        if self.playing:
            self._tick()

    def _tick(self):
        if not self.playing or self.df is None:
            return
        v = int(self.v_frame.get()) + 2
        if v >= len(self.df) - 1:
            v = len(self.df) - 1
            self.playing = False
            self.b_play.config(text="▶ play")
        self.v_frame.set(v)
        self.draw(v)
        if self.playing:
            self.root.after(40, self._tick)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    Viewer().run()
