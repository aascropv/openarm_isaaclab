#!/usr/bin/env python3
"""
replay_handover_demo.py — 純讀錄製的 HDF5 → 匯出 mp4（不需 Isaac、幾乎不吃 GPU）。

把收集的 handover demo 回放成影片：三相機並排（main / wrist_left / wrist_right）
+ 下方文字面板顯示逐幀「左右臂 16D 關節 + 夾爪」與「目標水果即時 xyz 座標」+ task 字串。

用法：
  # 三任務各匯出一筆代表 demo（最常用）
  python3 scripts/tools/replay_handover_demo.py --all

  # 指定任務 + 第幾筆
  python3 scripts/tools/replay_handover_demo.py --fruit banana --index 0

  # 直接指定 HDF5 檔
  python3 scripts/tools/replay_handover_demo.py --hdf5 datasets_handover/apple_xxx/demo_0003.hdf5
"""
import argparse
import glob
import os

import numpy as np
import h5py
import cv2
from PIL import Image, ImageDraw, ImageFont

DATA_ROOT  = "datasets_handover"
OUT_DIR    = "replay_videos"
JOINT_NAMES = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "Rgrip",
               "L1", "L2", "L3", "L4", "L5", "L6", "L7", "Lgrip"]
CAM_KEYS   = ["obs/cam_main", "obs/cam_wrist_left", "obs/cam_wrist_right"]
CAM_TITLES = ["main", "wrist_left", "wrist_right"]


def _font(size, bold=False):
    cands = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in cands:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def find_demo(fruit: str, index: int) -> str:
    files = []
    for d in sorted(glob.glob(f"{DATA_ROOT}/{fruit}_*")):
        files += sorted(glob.glob(f"{d}/demo_*.hdf5"))
    if not files:
        raise SystemExit(f"找不到 {fruit} 的 demo（{DATA_ROOT}/{fruit}_*）")
    if index >= len(files):
        raise SystemExit(f"{fruit} 只有 {len(files)} 筆，index={index} 超出")
    return files[index]


def render(hdf5_path: str, out_path: str, fps: int = 30) -> int:
    with h5py.File(hdf5_path, "r") as f:
        cams = [f[k][:] for k in CAM_KEYS]   # each (N,240,320,3)
        jp   = f["obs/joint_pos"][:]         # (N,16)
        cube = f["obs/cube_pos"][:]          # (N,3) 目標水果逐幀座標
        task = f.attrs.get("task", b"")
        fruit = f.attrs.get("fruit", b"")
        basket = f.attrs.get("basket", b"")
    task   = task.decode()   if isinstance(task, bytes)   else task
    fruit  = fruit.decode()  if isinstance(fruit, bytes)  else fruit
    basket = basket.decode() if isinstance(basket, bytes) else basket
    n = jp.shape[0]

    H, W = 240, 320
    panel_h = 250
    cw, ch = W * 3, H + panel_h
    fb, f0 = _font(14, bold=True), _font(13)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (cw, ch))

    for t in range(n):
        cv = Image.new("RGB", (cw, ch), (18, 18, 18))
        dr = ImageDraw.Draw(cv)
        # ── 三相機並排 ──
        for i, cam in enumerate(cams):
            cv.paste(Image.fromarray(cam[t][..., :3].astype(np.uint8)), (i * W, 0))
            dr.text((i * W + 5, 4), CAM_TITLES[i], fill=(255, 255, 0), font=fb)
        # ── 文字面板（三欄對齊三相機）──
        y = H + 6
        dr.text((8, y), f"task: {task}", fill=(0, 255, 180), font=fb)
        dr.text((8, y + 20), f"{fruit}->{basket}   frame {t+1:4d}/{n}", fill=(200, 200, 200), font=f0)
        # 欄1: 右臂 / 欄2: 左臂
        dr.text((8,     y + 46), "Right arm", fill=(255, 170, 170), font=fb)
        dr.text((W + 8, y + 46), "Left arm",  fill=(170, 170, 255), font=fb)
        for r in range(8):
            dr.text((8,     y + 66 + r * 19), f"{JOINT_NAMES[r]:6s}{jp[t][r]:+.4f}",   fill=(228, 228, 228), font=f0)
            dr.text((W + 8, y + 66 + r * 19), f"{JOINT_NAMES[8+r]:6s}{jp[t][8+r]:+.4f}", fill=(228, 228, 228), font=f0)
        # 欄3: 目標水果即時座標
        fx, fy, fz = cube[t]
        dr.text((2 * W + 8, y + 46), "target fruit xyz", fill=(255, 215, 110), font=fb)
        dr.text((2 * W + 8, y + 70),  f"x {fx:+.4f}", fill=(255, 215, 110), font=f0)
        dr.text((2 * W + 8, y + 90),  f"y {fy:+.4f}", fill=(255, 215, 110), font=f0)
        dr.text((2 * W + 8, y + 110), f"z {fz:+.4f}", fill=(255, 215, 110), font=f0)
        dr.text((2 * W + 8, y + 140), f"lifted maxZ {cube[:t+1,2].max():.3f}", fill=(180, 180, 180), font=f0)

        vw.write(cv2.cvtColor(np.asarray(cv), cv2.COLOR_RGB2BGR))
    vw.release()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruit", choices=["apple", "strawberry", "banana"])
    ap.add_argument("--index", type=int, default=0, help="第幾筆 demo（預設 0）")
    ap.add_argument("--hdf5", type=str, help="直接指定 HDF5 檔")
    ap.add_argument("--all", action="store_true", help="三任務各匯出一筆代表 demo")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    jobs = []
    if args.all:
        for fr in ["apple", "strawberry", "banana"]:
            jobs.append((find_demo(fr, args.index), f"{OUT_DIR}/{fr}_demo{args.index:04d}.mp4"))
    elif args.hdf5:
        base = os.path.basename(args.hdf5).replace(".hdf5", "")
        jobs.append((args.hdf5, f"{OUT_DIR}/{base}.mp4"))
    elif args.fruit:
        fp = find_demo(args.fruit, args.index)
        jobs.append((fp, f"{OUT_DIR}/{args.fruit}_demo{args.index:04d}.mp4"))
    else:
        raise SystemExit("請指定 --all 或 --fruit X 或 --hdf5 path")

    for src, out in jobs:
        n = render(src, out, args.fps)
        sz = os.path.getsize(out) / 1024 / 1024
        print(f"[OK] {src}\n     → {out}  ({n} 幀, {n/args.fps:.1f}s, {sz:.1f}MB)")


if __name__ == "__main__":
    main()
