#!/usr/bin/env python3
"""
Multi-episode benchmark orchestrator for ACT / SmolVLA policies.

Pipeline:
  Terminal 1: bash scripts/isaac/run_benchmark_scene.sh
  Terminal 2: bash policy_bridges/run_smolvla_rtc_bridge.sh  (或 run_act_bridge.sh)
  Terminal 3: python3 scripts/tools/benchmark_inference.py --policy smolvla --n_episodes 20

跟 Isaac scene + bridge 透過 /tmp/openarm_episode.json IPC：
  - orchestrator 寫 {"state": "reset", "episode": N}        → 觸發 Isaac reset
  - Isaac 完成 reset 後寫 {"state": "ready", "episode": N, "cube": [x,y,z]}
  - Isaac 每 0.5s 更新 {"state": "running", "cube": [x,y,z]}
  - orchestrator 偵測 cube_in_basket() 或超時 → 紀錄 + 下一個 episode

輸出：results/benchmark_{policy}_{timestamp}.csv
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── 設定（跟 collect_basket_demos.py / run_benchmark_scene.py 一致）────────

EPISODE_FILE = "/tmp/openarm_episode.json"

# Basket bounding box（用 Isaac 即時 basket pose 當中心，避免被撞開時誤判）
BASKET_HALF_X   = 0.105
BASKET_HALF_Y   = 0.070
# Z 上限：cube 必須在 basket rim 以下才算「進到籃子」
# basket bottom ~ 0.21, rim ~ 0.27（依 KLT_box 模型），cube 半徑 ~ 0.015
# 設 0.27 是 conservative：cube 邊緣低於 rim 才算 in
BASKET_RIM_Z    = 0.27

# 成功確認的時間視窗（cube 須在 basket 內 + z 穩定）
SUCCESS_CONFIRM_S       = 1.5   # cube 必須在 basket bbox 內持續多少秒
SUCCESS_Z_STABILITY_MM  = 5.0   # cube z 在最後 0.5s 變化必須 < 此值（mm）才算 settle

# 籃子初始位置（reset 後 snapshot 用，diff > 此值代表 basket 被撞開）
BASKET_DISPLACEMENT_TH = 0.03   # 3cm

# 預設 timeout 60s（實測 SmolVLA 最慢 ~49s）
DEFAULT_TIMEOUT_S = 60.0
SETTLE_WAIT_S     = 2.0   # reset 完成後等 settle（包含 bridge settle 1s + 物理 1s）
POLL_INTERVAL_S   = 0.5   # 多久 check 一次 cube 位置

# 成功後再給 policy 機會做「放開 cube + 收手」的時間
# 偵測 cube_in_basket 後，等手臂回 home 或最多 HOME_WAIT_S 秒
HOME_WAIT_S            = 8.0   # 最大寬限期
ARM_HOME_TOL_RAD       = 0.15  # 各 joint 距離 0 < tol 即視為 home (~8.6°)
GRIP_OPEN_TH           = 0.030 # gripper 須 >= 此值（打開）才算釋放


def cube_in_basket(cube: list, basket: list) -> bool:
    """方塊是否在 basket bounding box 內（含 Z 上限：必須在 rim 以下）。

    Bug 修正歷史：
      - 舊版只看 X/Y，導致「夾爪把 cube 舉到 basket 上方但還沒放」就誤判成功
      - 加 Z 上限 BASKET_RIM_Z 後，cube 必須在 basket 內（不能懸空）才算 in
    """
    if not cube or not basket:
        return False
    cx, cy, cz = cube
    bx, by, _ = basket
    in_x = abs(cx - bx) <= BASKET_HALF_X
    in_y = abs(cy - by) <= BASKET_HALF_Y
    in_z = cz < BASKET_RIM_Z  # cube 必須在 basket rim 以下（不能還在手裡）
    return in_x and in_y and in_z


def arm_at_home(left_joints: dict) -> bool:
    """左臂回到 home 的判定：
       - 只檢查 joint1-4（肩+肘，決定手掌空間位置）
       - 不檢查 joint5-7（wrist 旋轉），因為手掌可能卡平台導致 wrist 角度卡住，
         但手臂整體已收回；卡 wrist 不影響下一個 episode 起始狀態
       - gripper 必須打開（cube 已釋放）
    """
    arm_keys = [f"openarm_left_joint{i}" for i in range(1, 5)]  # j1-j4 only
    grip_key = "openarm_left_finger_joint1"
    for k in arm_keys:
        v = left_joints.get(k)
        if v is None or abs(v) > ARM_HOME_TOL_RAD:
            return False
    grip = left_joints.get(grip_key, 0.0)
    return grip >= GRIP_OPEN_TH


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
    """錯誤 case 也回傳完整 schema 的 dict，避免 DictWriter 欄位不符。"""
    return {
        "episode": ep_id,
        "success": False,
        "timeout": False,
        "duration_s": 0.0,
        "arm_home_wait_s": 0.0,
        "arm_returned_home": False,
        "basket_knocked": False,
        "basket_displacement_cm": 0.0,
        "fail_reason": reason,
        "cube_init_x": None, "cube_init_y": None, "cube_init_z": None,
        "cube_final_x": None, "cube_final_y": None, "cube_final_z": None,
        "basket_init_x": None, "basket_init_y": None,
        "basket_final_x": None, "basket_final_y": None,
    }


def run_single_episode(ep_id: int, timeout_s: float) -> dict:
    """跑一個 episode：reset → 等 settle → poll cube 直到成功 / timeout。返回紀錄 dict。"""
    print(f"\n{'='*60}")
    print(f"[EP {ep_id:03d}] Triggering reset...")

    # 1. 觸發 Isaac reset
    #    寫多次（每 0.5s 一次）避免 Isaac 剛啟動時 idle 寫覆蓋掉 reset signal（race condition）
    write_episode({"state": "reset", "episode": ep_id})

    # 2. 等 Isaac 把 state 改成 ready；期間若仍看到 idle / 沒看到 episode id 變化，重發 reset
    t0 = time.time()
    max_wait = 30.0
    ready_data = None
    while time.time() - t0 < max_wait:
        data = read_episode()
        state = data.get("state")
        ep_in_file = data.get("episode")
        if state in ("ready", "running") and ep_in_file == ep_id:
            ready_data = data
            break
        # 若 Isaac 還在 idle（沒處理 reset signal），重寫 reset
        if state == "idle":
            write_episode({"state": "reset", "episode": ep_id})
        time.sleep(0.5)
    if ready_data is None:
        print(f"[EP {ep_id:03d}] ERROR: Isaac never returned ready state")
        return empty_record(ep_id, "isaac_not_ready")
    cx0, cy0, cz0 = ready_data["cube"]
    basket0 = ready_data.get("basket")  # 可能不存在（舊 Isaac scene），fallback hardcoded
    if basket0 is None:
        basket0 = [0.298, 0.000, 0.246]
    bx0, by0, bz0 = basket0
    print(f"[EP {ep_id:03d}] Isaac ready. Cube={tuple(round(v,3) for v in (cx0,cy0,cz0))}, "
          f"basket={tuple(round(v,3) for v in (bx0,by0,bz0))}")

    # 3. 等 settle（bridge 自己 settle 1s + 餘裕）
    print(f"[EP {ep_id:03d}] Settling {SETTLE_WAIT_S:.1f}s before checking success...")
    time.sleep(SETTLE_WAIT_S)

    # 4. Poll 直到成功 / timeout
    t_start = time.time()
    success = False
    cx_last, cy_last, cz_last = cx0, cy0, cz0
    bx_last, by_last, bz_last = bx0, by0, bz0
    while time.time() - t_start < timeout_s:
        data = read_episode()
        cube = data.get("cube")
        basket = data.get("basket") or basket0  # fallback if Isaac 沒寫
        if cube and basket:
            cx_last, cy_last, cz_last = cube
            bx_last, by_last, bz_last = basket
            if cube_in_basket(cube, basket):
                # 持續 SUCCESS_CONFIRM_S 秒都在 basket 內，並且 z 穩定才算成功
                # 修正歷史:
                #   - 0.5s → 1.5s (cube 邊緣著陸滾落的時間)
                #   - 加 Z stability check (避免 cube 還在落下就算成功)
                success_confirm_t = time.time()
                still_in = True
                z_samples: list[float] = []
                while time.time() - success_confirm_t < SUCCESS_CONFIRM_S:
                    time.sleep(0.1)
                    d2 = read_episode()
                    c2 = d2.get("cube"); b2 = d2.get("basket") or basket0
                    if not c2 or not b2:
                        continue
                    if not cube_in_basket(c2, b2):
                        still_in = False
                        break
                    # 收集最後 0.5s 的 z 樣本給穩定性判定用
                    if time.time() - success_confirm_t >= (SUCCESS_CONFIRM_S - 0.5):
                        z_samples.append(c2[2])
                if still_in and len(z_samples) >= 3:
                    # Z stability check：最後 0.5s cube z 變化必須 < SUCCESS_Z_STABILITY_MM mm
                    z_range_mm = (max(z_samples) - min(z_samples)) * 1000.0
                    if z_range_mm < SUCCESS_Z_STABILITY_MM:
                        success = True
                        cx_last, cy_last, cz_last = cube
                        bx_last, by_last, bz_last = basket
                        break
                    else:
                        # cube 還在動 — 繼續 poll，下次循環再試
                        print(f"  [EP {ep_id:03d}] cube z 不穩 ±{z_range_mm:.0f}mm，繼續等 settle")
        time.sleep(POLL_INTERVAL_S)

    duration = time.time() - t_start
    timed_out = not success
    print(f"[EP {ep_id:03d}] {'✓ SUCCESS' if success else '✗ FAIL (timeout)'} "
          f"after {duration:.1f}s. Cube final = ({cx_last:.3f}, {cy_last:.3f}, {cz_last:.3f})")

    # 5. 成功後等 arm 回 home（或 grace timeout）才 reset
    # 不管成功失敗都等，因為失敗時 arm 也可能還在亂動，給 1s 緩衝就 reset
    arm_home_wait = 0.0
    arm_returned_home = False
    if success:
        t_home_start = time.time()
        print(f"[EP {ep_id:03d}] Waiting up to {HOME_WAIT_S:.0f}s for arm to return home...")
        while time.time() - t_home_start < HOME_WAIT_S:
            data = read_episode()
            left = data.get("left_joints", {})
            if left and arm_at_home(left):
                arm_returned_home = True
                break
            time.sleep(POLL_INTERVAL_S)
        arm_home_wait = round(time.time() - t_home_start, 1)
        if arm_returned_home:
            print(f"[EP {ep_id:03d}] Arm returned home after {arm_home_wait}s.")
        else:
            print(f"[EP {ep_id:03d}] Arm did NOT return home in {HOME_WAIT_S:.0f}s, proceeding anyway.")
    else:
        # 失敗時短暫緩衝，讓 robot 動作停下來再 reset
        time.sleep(1.0)

    # 籃子是否被撞開：與初始位置 XY 差距
    basket_displaced = ((bx_last - bx0) ** 2 + (by_last - by0) ** 2) ** 0.5
    basket_was_knocked = basket_displaced > BASKET_DISPLACEMENT_TH
    if basket_was_knocked:
        print(f"[EP {ep_id:03d}] WARN: basket displaced {basket_displaced*100:.1f}cm during episode "
              f"(init={bx0:.3f},{by0:.3f} → final={bx_last:.3f},{by_last:.3f})")

    return {
        "episode": ep_id,
        "success": success,
        "timeout": timed_out,
        "duration_s": round(duration, 2),
        "arm_home_wait_s": arm_home_wait,
        "arm_returned_home": arm_returned_home,
        "basket_knocked": basket_was_knocked,
        "basket_displacement_cm": round(basket_displaced * 100, 1),
        "fail_reason": "timeout" if timed_out else "",
        "cube_init_x": round(cx0, 4), "cube_init_y": round(cy0, 4), "cube_init_z": round(cz0, 4),
        "cube_final_x": round(cx_last, 4), "cube_final_y": round(cy_last, 4), "cube_final_z": round(cz_last, 4),
        "basket_init_x": round(bx0, 4), "basket_init_y": round(by0, 4),
        "basket_final_x": round(bx_last, 4), "basket_final_y": round(by_last, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["act", "smolvla", "pi05", "pi05_rtc"], required=True,
                        help="policy 名稱（純標記用，bridge 需自己啟動）")
    parser.add_argument("--n_episodes", type=int, default=20,
                        help="跑幾個 episode（從 start_episode 開始算）")
    parser.add_argument("--start_episode", type=int, default=0,
                        help="從 ep N 開始跑（partial rerun 用，例如 --start_episode 5 --n_episodes 1 只跑 ep5）")
    parser.add_argument("--seed", type=int, default=42,
                        help="cube seed（純 log 用，要實際生效須在 Isaac scene 啟動時用同 --seed）")
    parser.add_argument("--timeout_s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    # 確認 Isaac scene 已啟動（EPISODE_FILE 至少存在）
    if not Path(EPISODE_FILE).exists():
        print(f"ERROR: {EPISODE_FILE} does not exist. Please start Isaac benchmark scene first:")
        print("  Terminal 1: bash scripts/isaac/run_benchmark_scene.sh")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"benchmark_{args.policy}_seed{args.seed}_{ts}.csv"
    end_episode = args.start_episode + args.n_episodes
    print(f"\n[BENCHMARK] policy={args.policy}, seed={args.seed}, "
          f"episodes={args.start_episode}..{end_episode - 1} ({args.n_episodes} eps), "
          f"timeout={args.timeout_s}s")
    print(f"[BENCHMARK] Output: {csv_path}\n")
    print(f"[BENCHMARK] ⚠️ 請確認 Isaac scene 啟動時用了 --seed {args.seed}（否則 cube 位置不會對應）\n")

    # 顯式 fieldnames（避免 error case dict 缺欄位導致 DictWriter 報錯）
    fieldnames = list(empty_record(0, "").keys())

    results: list[dict] = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(args.start_episode, end_episode):
            rec = run_single_episode(i, args.timeout_s)
            results.append(rec)
            writer.writerow(rec)
            f.flush()

    # Summary
    n = len(results)
    n_success = sum(1 for r in results if r["success"])
    avg_dur = sum(r["duration_s"] for r in results) / n
    avg_dur_success = (
        sum(r["duration_s"] for r in results if r["success"]) / max(n_success, 1)
    )
    print(f"\n{'='*60}")
    print(f"[BENCHMARK COMPLETE] policy={args.policy}")
    print(f"  Success rate:      {n_success}/{n} = {100*n_success/n:.0f}%")
    print(f"  Avg duration:      {avg_dur:.1f}s")
    print(f"  Avg success time:  {avg_dur_success:.1f}s")
    print(f"  CSV saved:         {csv_path}")
    print(f"{'='*60}\n")

    # 讓 Isaac 回 idle
    write_episode({"state": "idle"})


if __name__ == "__main__":
    main()
