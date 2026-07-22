#!/usr/bin/env python3
"""
Handover demo 事後篩選（收集腳本 success 只看入籃，會混進 move_home 失敗/甩飛）。

套 5 標準把 raw demos 篩成乾淨集，壞的搬到 <dir>/excluded/，多的搬到 <dir>/spare/。
判準閾值由 datasets_handover/banana_* 實測校準（見 project_apple_v2_dataset 記憶）：
  1. success == True
  2. obj_final 落在目標籃 bbox
  3. 起步對齊：frame0 手臂在 home（|arm| < HOME_TOL）
  4. 回 home：末幀手臂相對 frame0 位移 < RETURN_TOL（抓 move_home OMPL 偶發失敗）
  5. 無甩飛：cube_pos z 全程 max < FLING_MAXZ

用法：
  python3 scripts/tools/filter_fruit_demos.py --fruit banana --basket yellow \
      --dir datasets_handover/banana_2026-07-01_23-34-06 --cap 100
"""
import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

BASKET_BBOX = {
    "red":    dict(xmin=0.1983, xmax=0.3486, ymin=-0.1154, ymax=-0.0306),
    "yellow": dict(xmin=0.1983, xmax=0.3486, ymin=-0.2210, ymax=-0.1361),
}
ARM_IDX = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]
HOME_TOL = 0.05      # frame0 手臂 max|joint| 容忍（reset home ≈ 0）
RETURN_TOL = 0.30    # 末幀相對 frame0 位移容忍（回 home）
FLING_MAXZ = 0.60    # cube z 上限（正常 handover 舉到 ~0.50；甩飛 >0.6）
RSPIN_TOL = 2.5      # 右臂 base(R_j1=idx0)全程轉幅上限（乾淨 <1.9；OMPL 甩大圈繞遠路 ~3.5 → 剔除）


def check(path: Path, basket: str) -> str | None:
    """回傳失敗原因字串；通過回 None。"""
    with h5py.File(path, "r") as f:
        if not bool(f.attrs.get("success", False)):
            return "not_success"
        of = np.asarray(f.attrs["obj_final"], dtype=float)
        jp = f["obs/joint_pos"][:]
        cz = f["obs/cube_pos"][:, 2]
    b = BASKET_BBOX[basket]
    if not (b["xmin"] <= of[0] <= b["xmax"] and b["ymin"] <= of[1] <= b["ymax"]):
        return f"not_in_basket(x={of[0]:.3f},y={of[1]:.3f})"
    f0 = np.abs(jp[0, ARM_IDX]).max()
    if f0 > HOME_TOL:
        return f"bad_start(arm={f0:.3f})"
    dev = np.abs(jp[-1, ARM_IDX] - jp[0, ARM_IDX]).max()
    if dev > RETURN_TOL:
        return f"no_return_home(dev={dev:.3f})"
    mz = float(cz.max())
    if mz > FLING_MAXZ:
        return f"fling(maxz={mz:.3f})"
    rj1 = float(jp[:, 0].max() - jp[:, 0].min())   # 右臂 base 轉幅
    if rj1 > RSPIN_TOL:
        return f"arm_spin(rj1={rj1:.2f})"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fruit", required=True)
    ap.add_argument("--basket", required=True, choices=["red", "yellow"])
    ap.add_argument("--dir", required=True, nargs="+", help="一個或多個 raw 目錄")
    ap.add_argument("--cap", type=int, default=0, help=">0 時把超出的乾淨集搬到 spare/")
    ap.add_argument("--apply", action="store_true", help="實際搬檔（預設 dry-run 只報告）")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.dir]
    files = []
    for d in dirs:
        files.extend(sorted(d.glob("demo_*.hdf5")))
    print(f"掃描 {len(files)} 個 demo（{[str(d) for d in dirs]}）")

    clean, bad = [], []
    for p in files:
        reason = check(p, args.basket)
        (bad if reason else clean).append((p, reason))

    print(f"\n== 篩選結果：乾淨 {len(clean)} / 剔除 {len(bad)} ==")
    for p, r in bad:
        print(f"  [剔除] {p.parent.name}/{p.name}: {r}")

    # 超出 cap 的乾淨集標記為 spare（保留 frame 順序，砍最後幾筆）
    spare = []
    if args.cap and len(clean) > args.cap:
        spare = clean[args.cap:]
        clean = clean[:args.cap]
        print(f"\n== cap={args.cap}：{len(spare)} 筆多餘乾淨集 → spare/ ==")

    if not args.apply:
        print(f"\n[dry-run] 乾淨保留 {len(clean)} 筆。加 --apply 實際搬檔。")
        return

    for p, r in bad:
        dst = p.parent / "excluded"
        dst.mkdir(exist_ok=True)
        shutil.move(str(p), str(dst / p.name))
    for p, _ in spare:
        dst = p.parent / "spare"
        dst.mkdir(exist_ok=True)
        shutil.move(str(p), str(dst / p.name))
    print(f"\n[apply] 剔除 {len(bad)} → excluded/，spare {len(spare)} → spare/，乾淨保留 {len(clean)} 筆。")


if __name__ == "__main__":
    main()
