#!/usr/bin/env python3
"""eval_viewer.py — 逐幀檢視 eval 測試過程（獨立 Python GUI，非 notebook）。

顯示：三相機並排影像(overhead|left|right) + 16 關節值 + 雙臂側視(x-z 看卡不卡平台) + 該集入籃/回home。
資料來自 eval_record_view.sh 產生的：<base>.csv(EEF frame log 含 wall) + <base>_3cam.mp4(+.timestamps.csv)。
影像用 wall 絕對時間跟 log frame 對齊。

用法:
  conda run -n lerobot python3 scripts/tools/eval_viewer.py \
      --base outputs/evalview_apple_v2_040000  [--fruit apple --basket red]
  （--base 給共同前綴，會自動找 .csv / _3cam.mp4）
"""
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle

RB = dict(x=(0.198, 0.349), y=(-0.115, -0.031)); YB = dict(x=(0.198, 0.349), y=(-0.221, -0.136))
PLAT = dict(x=(0.166, 0.568), top=0.193); BASK = {"red": RB, "yellow": YB}


def arm_links(df, arm):
    order = [f"openarm_{arm}_link{i}" for i in range(1, 8)] + [f"openarm_{arm}_left_finger", f"openarm_{arm}_right_finger"]
    return [b for b in order if f"{b}_x" in df.columns]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="共同前綴（<base>.csv + <base>_3cam.mp4）")
    ap.add_argument("--log", default=None); ap.add_argument("--mp4", default=None)
    ap.add_argument("--fruit", default="apple"); ap.add_argument("--basket", default="red")
    a = ap.parse_args()
    log = a.log or (a.base + ".csv")
    mp4 = a.mp4 or (a.base + "_3cam.mp4")

    df = pd.read_csv(log)
    df = df[df.state.isin(["ready", "running"])].reset_index(drop=True)
    JOINTS = [c for c in df.columns if c.startswith("openarm_") and "joint" in c and not c.endswith(("_x", "_y", "_z"))]
    RIGHT, LEFT = arm_links(df, "right"), arm_links(df, "left")
    FRUITS = [f for f in ["apple", "strawberry", "banana"] if f"{f}_x" in df.columns]
    episodes = sorted(df.episode.unique())
    ep_start = {ep: df.index[df.episode == ep][0] for ep in episodes}

    # 權威狀態：讀 benchmark result CSV（有 success + arm_returned_home，8s 窗口正確偵測）。
    # ★ 不能只看 log 末幀——手臂回 home 後 policy 常又動、末幀已離 home，會誤判 False。
    import glob
    ckpt = a.base.rstrip("/").split("_")[-1]
    bench = {}
    bfiles = sorted(glob.glob(f"results_handover_eval/handover_{a.fruit}_{a.basket}_recview_{ckpt}_*.csv"))
    if bfiles:
        for _, br in pd.read_csv(bfiles[-1]).iterrows():
            bench[int(br["episode"])] = (str(br["success"]).strip().lower() == "true",
                                         str(br["arm_returned_home"]).strip().lower() == "true")

    def ep_status(ep):
        if int(ep) in bench:                       # 用 benchmark 權威值
            return bench[int(ep)]
        e = df[df.episode == ep]; last = e.iloc[-1]; b = BASK[a.basket]  # fallback
        fx, fy, fz = last[f"{a.fruit}_x"], last[f"{a.fruit}_y"], last[f"{a.fruit}_z"]
        inb = b["x"][0] <= fx <= b["x"][1] and b["y"][0] <= fy <= b["y"][1] and fz < 0.28
        arms = [max(abs(row[j]) for j in JOINTS if "finger" not in j) for _, row in e.tail(30).iterrows()]
        return inb, (min(arms) < 0.20 if arms else False)   # 末30幀曾回 home

    # 三相機 mp4 + timestamps
    cap = cv2.VideoCapture(mp4)
    try:
        ts = pd.read_csv(mp4 + ".timestamps.csv")
        ts_wall = ts["wall"].values
    except Exception:
        ts_wall = None
    def frame_img(row_wall):
        if ts_wall is None or not cap.isOpened():
            return None
        k = int(np.argmin(np.abs(ts_wall - row_wall)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, im = cap.read()
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if ok else None

    # ── figure ──
    fig = plt.figure(figsize=(15, 8)); fig.canvas.manager.set_window_title(f"eval viewer — {a.base}")
    ax_img = fig.add_axes([0.03, 0.42, 0.94, 0.52]); ax_img.axis("off")
    ax_side = fig.add_axes([0.06, 0.10, 0.40, 0.26])
    ax_j = fig.add_axes([0.55, 0.10, 0.40, 0.26])
    ax_sl = fig.add_axes([0.07, 0.03, 0.52, 0.03])
    slider = Slider(ax_sl, "frame", 0, len(df) - 1, valinit=0, valstep=1)
    bplay = Button(fig.add_axes([0.62, 0.03, 0.10, 0.04]), "▶ play")
    bprev = Button(fig.add_axes([0.76, 0.03, 0.08, 0.04]), "◀ ep")
    bnext = Button(fig.add_axes([0.86, 0.03, 0.08, 0.04]), "ep ▶")

    def draw(i):
        i = int(i); r = df.iloc[i]
        ep, fr = int(r.episode), int(r.frame); inb, home = ep_status(ep)
        # 影像
        img = frame_img(r.wall) if "wall" in df.columns else None
        ax_img.clear(); ax_img.axis("off")
        if img is not None:
            ax_img.imshow(img)
            w = img.shape[1] // 3
            for j, lab in enumerate(["overhead", "wrist_left", "wrist_right"]):
                ax_img.text(w*j + w/2, -8, lab, ha="center", fontsize=10)
        else:
            ax_img.text(0.5, 0.5, "(無影像 / mp4 對不上)", ha="center", va="center", transform=ax_img.transAxes)
        # 側視 x-z
        ax_side.clear()
        for links, c in [(RIGHT, "tab:blue"), (LEFT, "tab:orange")]:
            ax_side.plot([r[f"{b}_x"] for b in links], [r[f"{b}_z"] for b in links], "-o", color=c, ms=3)
        for f in FRUITS:
            ax_side.scatter([r[f"{f}_x"]], [r[f"{f}_z"]], s=90, marker="*", zorder=5)
        ax_side.add_patch(Rectangle((PLAT["x"][0], 0), PLAT["x"][1]-PLAT["x"][0], PLAT["top"], color="brown", alpha=.18))
        ax_side.axhline(PLAT["top"], ls=":", c="brown")
        ax_side.set(xlabel="x", ylabel="z", title="side x-z (platform hit?)", xlim=(-0.1, 0.7), ylim=(0, 0.65)); ax_side.grid(alpha=.3)
        # 關節
        ax_j.clear()
        names = [j.replace("openarm_", "").replace("_joint", "_j").replace("_finger_j1", "_grip") for j in JOINTS]
        ax_j.barh(range(len(JOINTS)), [r[j] for j in JOINTS],
                  color=["tab:blue" if "right" in j else "tab:orange" for j in JOINTS])
        ax_j.set_yticks(range(len(JOINTS))); ax_j.set_yticklabels(names, fontsize=6)
        ax_j.axvline(0, c="k", lw=.5); ax_j.invert_yaxis(); ax_j.set(xlabel="rad", title="16 joints", xlim=(-2.2, 2.2)); ax_j.grid(alpha=.3, axis="x")
        fig.suptitle(f"ep{ep}  frame {fr}   |   in-basket={'YES' if inb else 'NO'}   arm-home={'YES' if home else 'NO'}"
                     f"   (blue=right arm, orange=left)", fontsize=13, fontweight="bold")
        fig.canvas.draw_idle()

    slider.on_changed(draw)
    def jump(delta):
        cur = int(slider.val); ep = int(df.iloc[cur].episode)
        idx = episodes.index(ep) + delta
        if 0 <= idx < len(episodes):
            slider.set_val(ep_start[episodes[idx]])
    bprev.on_clicked(lambda e: jump(-1)); bnext.on_clicked(lambda e: jump(+1))

    # ── play / pause（timer 自動推進 slider）──
    st = {"play": False, "step": 2}
    timer = fig.canvas.new_timer(interval=40)
    def _tick():
        if not st["play"]:
            return
        v = int(slider.val) + st["step"]
        if v >= len(df) - 1:
            v = len(df) - 1
            st["play"] = False; bplay.label.set_text("▶ play"); timer.stop()
        slider.set_val(v)
    timer.add_callback(_tick)
    def toggle(_e):
        st["play"] = not st["play"]
        if st["play"]:
            if int(slider.val) >= len(df) - 1:
                slider.set_val(0)
            bplay.label.set_text("⏸ pause"); timer.start()
        else:
            bplay.label.set_text("▶ play"); timer.stop()
    bplay.on_clicked(toggle)

    draw(0)
    print(f"episodes: {episodes}")
    for ep in episodes:
        inb, home = ep_status(ep); print(f"  ep{ep}: 入籃={inb} 回home={home}")
    plt.show()


if __name__ == "__main__":
    main()
