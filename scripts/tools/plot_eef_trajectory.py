#!/usr/bin/env python3
"""plot_eef_trajectory.py ─ 把 record_eef_trajectory.py 存的 CSV 畫成 EEF 3D 軌跡圖。

左臂藍、右臂紅，顏色由淺到深 = 時間先後（看得出去程/回程順序）。
另加一張 side view (X-Z) + top view (X-Y) + 左臂 roll 隨時間（看交接有沒有轉到水平 roll≈±π/2）。
用有 matplotlib 的 env 跑：
  conda run -n lerobot python3 scripts/tools/plot_eef_trajectory.py outputs/eef_traj.csv
輸出：同名 .png
"""
import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 中文標題不變框框：註冊系統 Noto CJK 字型（.ttc 內含 JP/SC/TC 全 CJK glyph）
_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    font_manager.fontManager.addfont(_CJK)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_CJK).get_name()
except Exception:
    pass
plt.rcParams["axes.unicode_minus"] = False   # 避免負號也變框框


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    d = {}
    for k in rows[0]:
        d[k] = [float(r[k]) if r[k] != "" else math.nan for r in rows]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.splitext(a.csv)[0] + ".png"
    d = load(a.csv)
    t = d["t"]

    fig = plt.figure(figsize=(16, 10))

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    for side, cmap in (("left", "Blues"), ("right", "Reds")):
        x, y, z = d[f"{side}_x"], d[f"{side}_y"], d[f"{side}_z"]
        ax3d.scatter(x, y, z, c=t, cmap=cmap, s=6)
        ax3d.plot(x, y, z, color="gray", lw=0.3, alpha=0.5)
        ax3d.text(x[0], y[0], z[0], f"{side} start", fontsize=7)
        ax3d.text(x[-1], y[-1], z[-1], f"{side} end", fontsize=7)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title("3D EEF path (淺→深 = 時間)")

    axxz = fig.add_subplot(2, 2, 2)
    for side, c in (("left", "tab:blue"), ("right", "tab:red")):
        axxz.scatter(d[f"{side}_x"], d[f"{side}_z"], c=t, cmap="viridis", s=6)
        axxz.plot(d[f"{side}_x"], d[f"{side}_z"], color=c, lw=0.4, alpha=0.6, label=side)
    axxz.set_xlabel("X"); axxz.set_ylabel("Z"); axxz.set_title("Side view X-Z")
    axxz.legend(); axxz.grid(alpha=0.3)

    axxy = fig.add_subplot(2, 2, 3)
    for side, c in (("left", "tab:blue"), ("right", "tab:red")):
        axxy.scatter(d[f"{side}_x"], d[f"{side}_y"], c=t, cmap="viridis", s=6)
        axxy.plot(d[f"{side}_x"], d[f"{side}_y"], color=c, lw=0.4, alpha=0.6, label=side)
    axxy.set_xlabel("X"); axxy.set_ylabel("Y"); axxy.set_title("Top view X-Y")
    axxy.legend(); axxy.grid(alpha=0.3)

    axr = fig.add_subplot(2, 2, 4)
    for side, c in (("left", "tab:blue"), ("right", "tab:red")):
        axr.plot(t, [math.degrees(v) for v in d[f"{side}_roll"]], color=c, label=f"{side} roll")
    axr.axhline(90, color="green", ls="--", lw=0.8, label="±90° (水平交接)")
    axr.axhline(-90, color="green", ls="--", lw=0.8)
    axr.axhline(180, color="orange", ls=":", lw=0.8, label="±180° (top-down)")
    axr.axhline(-180, color="orange", ls=":", lw=0.8)
    axr.set_xlabel("t (s)"); axr.set_ylabel("roll (deg)")
    axr.set_title("EEF roll vs time (交接應到 ±90°)")
    axr.legend(fontsize=7); axr.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"saved -> {out}  ({len(t)} samples, {t[-1]:.1f}s)")


if __name__ == "__main__":
    main()
