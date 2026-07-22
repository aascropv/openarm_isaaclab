#!/usr/bin/env python3
"""demo_to_video.py ─ 把 collect_handover_demos 存的 demo HDF5 轉成可回看的 mp4。

三相機並排（main | wrist_left | wrist_right），fps 取自 HDF5 attrs、每幀左上角標 demo 名+幀號。
用有 cv2 的 env 跑（系統 python3 沒有）：
  單筆: conda run -n lerobot python3 scripts/tools/demo_to_video.py datasets_handover/apple_XXX/demo_0000.hdf5
  整批: conda run -n lerobot python3 scripts/tools/demo_to_video.py datasets_handover/apple_XXX/
        → 在該資料夾的 videos/ 下每筆一支 mp4
"""
import argparse
import glob
import os

import cv2
import h5py
import numpy as np

CAMS = ["cam_main", "cam_wrist_left", "cam_wrist_right"]


def render(h5path, out, cams):
    with h5py.File(h5path, "r") as f:
        fps = int(f.attrs.get("fps", 30))
        obs = f["obs"]
        arrs = [obs[c][:] for c in cams if c in obs]
        if not arrs:
            print(f"  [skip] {os.path.basename(h5path)}: 無相機影格")
            return
        n = min(a.shape[0] for a in arrs)
        frames = np.concatenate([a[:n] for a in arrs], axis=2)  # 沿寬度並排
    label = os.path.basename(h5path)
    H, W = frames.shape[1], frames.shape[2]
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(frames.shape[0]):
        bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, f"{label}  f{i}", (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        vw.write(bgr)
    vw.release()
    print(f"  saved {out}  ({frames.shape[0]} frames @ {fps}fps)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="單筆 demo.hdf5 或含 demo_*.hdf5 的資料夾")
    ap.add_argument("--out", default=None, help="輸出檔/資料夾（預設：單筆同名 .mp4；整批放 videos/）")
    ap.add_argument("--cams", default="all", choices=["all", "main"],
                    help="all=三相機並排；main=只有主相機")
    a = ap.parse_args()
    cams = CAMS if a.cams == "all" else ["cam_main"]

    if os.path.isdir(a.input):
        demos = sorted(glob.glob(os.path.join(a.input, "demo_*.hdf5")))
        vdir = a.out or os.path.join(a.input, "videos")
        os.makedirs(vdir, exist_ok=True)
        print(f"{len(demos)} demos → {vdir}")
        for d in demos:
            render(d, os.path.join(vdir, os.path.basename(d).replace(".hdf5", ".mp4")), cams)
    else:
        render(a.input, a.out or a.input.replace(".hdf5", ".mp4"), cams)


if __name__ == "__main__":
    main()
