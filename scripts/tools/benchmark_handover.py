#!/usr/bin/env python3
"""
benchmark_handover.py — handover policy 評測 orchestrator（單一任務）

驅動 run_handover_eval_scene.py：每集 reset（固定位置）→ 等 ready → poll 目標水果是否進目標籃
（XY 在籃內 + Z 低於 rim = 真的放進去，不是舉在交接高度）→ 紀錄成功率到 CSV。

三終端：
  T1: bash scripts/isaac/run_handover_eval_scene.sh
  T2: bash policy_bridges/run_smolvla_rtc_bridge.sh --ckpt <ckpt> --task "<task 字串>" --disable_gripper_snap
  T3: python3 scripts/tools/benchmark_handover.py --fruit banana --basket yellow --n_episodes 10

★ T2 的 --task 必須跟 --fruit/--basket 對應的 dataset task 字串完全一致。
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

EPISODE_FILE = "/tmp/openarm_episode.json"

# 籃子 footprint 半徑（由收集時 BASKET_BBOX 推：寬 0.150→half 0.075、高 0.085→half 0.042，略放寬）
BASKET_HALF_X = 0.078
BASKET_HALF_Y = 0.048
# 水果中心須低於此高度才算「放進籃」（交接高度 ~0.33-0.37，籃內 ~0.21-0.24 → 0.28 乾淨切開）
BASKET_RIM_Z  = 0.28

SUCCESS_CONFIRM_S      = 1.5    # 須持續在籃內這麼久
SUCCESS_Z_STABILITY_MM = 8.0    # 最後 0.5s 水果 z 變化 < 此值才算 settle
DEFAULT_TIMEOUT_S = 75.0        # handover 比 basket 長
SETTLE_WAIT_S     = 2.0
POLL_INTERVAL_S   = 0.5
HOME_WAIT_S       = 8.0
ARM_HOME_TOL      = 0.25        # 14 臂關節離 0 的容忍


def fruit_in_basket(fruit: list, basket: list) -> bool:
    if not fruit or not basket:
        return False
    fx, fy, fz = fruit
    bx, by, _ = basket
    in_xy = abs(fx - bx) <= BASKET_HALF_X and abs(fy - by) <= BASKET_HALF_Y
    return in_xy and (fz < BASKET_RIM_Z)


def arms_at_home(joints: dict) -> bool:
    if not joints:
        return False
    for jn, v in joints.items():
        if jn.endswith("finger_joint1") or jn.endswith("finger_joint2"):
            continue
        if abs(v) > ARM_HOME_TOL:
            return False
    return True


def read_episode() -> dict:
    try:
        with open(EPISODE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"state": "idle"}


def write_episode(payload: dict):
    tmp = EPISODE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, EPISODE_FILE)


def empty_record(ep_id: int, reason: str) -> dict:
    return {
        "episode": ep_id, "success": False, "timeout": False, "duration_s": 0.0,
        "arm_returned_home": False, "fail_reason": reason,
        "fruit_init_x": None, "fruit_init_y": None, "fruit_init_z": None,
        "fruit_final_x": None, "fruit_final_y": None, "fruit_final_z": None,
    }


def run_single_episode(ep_id: int, fruit: str, basket: str, timeout_s: float,
                       home_wait_s: float = HOME_WAIT_S) -> dict:
    print(f"\n{'='*60}\n[EP {ep_id:03d}] Triggering reset...")
    write_episode({"state": "reset", "episode": ep_id})

    t0 = time.time()
    ready = None
    while time.time() - t0 < 30.0:
        d = read_episode()
        if d.get("state") in ("ready", "running") and d.get("episode") == ep_id:
            ready = d
            break
        if d.get("state") == "idle":
            write_episode({"state": "reset", "episode": ep_id})
        time.sleep(0.5)
    if ready is None:
        print(f"[EP {ep_id:03d}] ERROR: Isaac never returned ready")
        return empty_record(ep_id, "isaac_not_ready")

    f0 = ready.get("fruits", {}).get(fruit)
    if not f0:
        return empty_record(ep_id, "no_fruit_pose")
    fx0, fy0, fz0 = f0
    print(f"[EP {ep_id:03d}] Ready. {fruit} init=({fx0:.3f},{fy0:.3f},{fz0:.3f}), target basket={basket}")

    print(f"[EP {ep_id:03d}] Settling {SETTLE_WAIT_S:.1f}s...")
    time.sleep(SETTLE_WAIT_S)

    t_start = time.time()
    success = False
    fx_last, fy_last, fz_last = fx0, fy0, fz0
    while time.time() - t_start < timeout_s:
        d = read_episode()
        fr = d.get("fruits", {}).get(fruit)
        bk = d.get("baskets", {}).get(basket)
        if fr and bk:
            fx_last, fy_last, fz_last = fr
            if fruit_in_basket(fr, bk):
                # 確認窗口：持續在籃內 + z 穩定
                tc = time.time()
                still_in = True
                z_samp = []
                while time.time() - tc < SUCCESS_CONFIRM_S:
                    time.sleep(0.1)
                    d2 = read_episode()
                    fr2 = d2.get("fruits", {}).get(fruit)
                    bk2 = d2.get("baskets", {}).get(basket) or bk
                    if not fr2:
                        continue
                    if not fruit_in_basket(fr2, bk2):
                        still_in = False
                        break
                    if time.time() - tc >= (SUCCESS_CONFIRM_S - 0.5):
                        z_samp.append(fr2[2])
                if still_in and len(z_samp) >= 3:
                    z_range_mm = (max(z_samp) - min(z_samp)) * 1000.0
                    if z_range_mm < SUCCESS_Z_STABILITY_MM:
                        success = True
                        fx_last, fy_last, fz_last = fr
                        break
                    else:
                        print(f"  [EP {ep_id:03d}] {fruit} z 不穩 ±{z_range_mm:.0f}mm，繼續等")
        time.sleep(POLL_INTERVAL_S)

    duration = time.time() - t_start
    timed_out = not success
    print(f"[EP {ep_id:03d}] {'✓ SUCCESS' if success else '✗ FAIL (timeout)'} after {duration:.1f}s. "
          f"{fruit} final=({fx_last:.3f},{fy_last:.3f},{fz_last:.3f})")

    arm_home = False
    if success:
        th = time.time()
        while time.time() - th < home_wait_s:
            if arms_at_home(read_episode().get("joints", {})):
                arm_home = True
                break
            time.sleep(POLL_INTERVAL_S)
        print(f"[EP {ep_id:03d}] arm returned home: {arm_home}")
    else:
        time.sleep(1.0)

    return {
        "episode": ep_id, "success": success, "timeout": timed_out,
        "duration_s": round(duration, 2), "arm_returned_home": arm_home,
        "fail_reason": "timeout" if timed_out else "",
        "fruit_init_x": round(fx0, 4), "fruit_init_y": round(fy0, 4), "fruit_init_z": round(fz0, 4),
        "fruit_final_x": round(fx_last, 4), "fruit_final_y": round(fy_last, 4), "fruit_final_z": round(fz_last, 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fruit", choices=["apple", "strawberry", "banana"], required=True)
    p.add_argument("--basket", choices=["red", "yellow"], required=True)
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--start_episode", type=int, default=0)
    p.add_argument("--timeout_s", type=float, default=DEFAULT_TIMEOUT_S)
    p.add_argument("--home_wait_s", type=float, default=HOME_WAIT_S,
                   help=f"入籃成功後等雙臂回 home 的秒數（default={HOME_WAIT_S}）。"
                        f"回程軌跡 ~10s+，要衡量『手臂回位』須拉大（如 25）")
    p.add_argument("--tag", type=str, default="", help="輸出檔名標記（如 ckpt30k）")
    p.add_argument("--output_dir", type=str, default="results_handover_eval")
    args = p.parse_args()

    if not Path(EPISODE_FILE).exists():
        print(f"ERROR: {EPISODE_FILE} 不存在，請先啟動 Isaac eval scene")
        sys.exit(1)

    out = Path(args.output_dir); out.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    csv_path = out / f"handover_{args.fruit}_{args.basket}{tag}_{ts}.csv"
    end = args.start_episode + args.n_episodes
    print(f"\n[BENCHMARK] fruit={args.fruit}→{args.basket}, eps={args.start_episode}..{end-1}, timeout={args.timeout_s}s")
    print(f"[BENCHMARK] Output: {csv_path}")
    print(f"[BENCHMARK] ⚠️ 確認 bridge 的 --task 對應 {args.fruit}（語言字串要跟 dataset 一致）\n")

    fieldnames = list(empty_record(0, "").keys())
    results = []
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(args.start_episode, end):
            rec = run_single_episode(i, args.fruit, args.basket, args.timeout_s, args.home_wait_s)
            results.append(rec)
            w.writerow(rec)
            f.flush()

    n = len(results)
    ns = sum(1 for r in results if r["success"])
    print(f"\n{'='*60}")
    print(f"[DONE] {args.fruit}→{args.basket}: {ns}/{n} = {100*ns/max(n,1):.0f}%")
    print(f"  CSV: {csv_path}")
    print(f"{'='*60}\n")
    write_episode({"state": "idle"})


if __name__ == "__main__":
    main()
