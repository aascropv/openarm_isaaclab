#!/usr/bin/env python3
"""
HDF5 demo dataset → LeRobotDataset（含 stats zero-variance patch）

從 `collect_basket_demos*.py` 產出的 HDF5 episodes，轉成 lerobot 訓練用的 LeRobotDataset，
並自動 patch right-arm zero-variance 的 stats，避免訓練 loss 爆 1e19。

用法：
  conda run -n lerobot python3 scripts/tools/hdf5_to_lerobot_dataset.py \
      --hdf5_dir ./datasets_basket_fixed_grid/2026-05-21_10-14-45 \
      --dataset_name basket_task_fixed_grid

  # 多個 run 合併：
  --hdf5_dir ./datasets_basket_fixed_grid/run1 ./datasets_basket_fixed_grid/run2

預設輸出到 `~/openarm_isaaclab/datasets_lerobot/openarm/<dataset_name>/`。
"""
import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# v2 標準 16D joint 順序（lerobot/folding 風格：R1..R7+R_grip, L1..L7+L_grip）
FOLDING_16D_NAMES = [
    "right_joint_1.pos", "right_joint_2.pos", "right_joint_3.pos", "right_joint_4.pos",
    "right_joint_5.pos", "right_joint_6.pos", "right_joint_7.pos", "right_gripper.pos",
    "left_joint_1.pos",  "left_joint_2.pos",  "left_joint_3.pos",  "left_joint_4.pos",
    "left_joint_5.pos",  "left_joint_6.pos",  "left_joint_7.pos",  "left_gripper.pos",
]

# Zero-variance patch 閾值（跟 policy_bridges/{pi05,smolvla_rtc}_bridge_node.py 同步）
MIN_NORM_RANGE = 0.01


def collect_hdf5_files(hdf5_dirs: list[Path], only_success: bool) -> list[Path]:
    files = []
    for d in hdf5_dirs:
        files.extend(sorted(d.glob("demo_*.hdf5")))
    print(f"找到 {len(files)} 個 HDF5 episodes")

    if only_success:
        kept = []
        for p in files:
            with h5py.File(p, "r") as f:
                if bool(f.attrs.get("success", False)):
                    kept.append(p)
        print(f"  保留 success=True：{len(kept)} / {len(files)}")
        files = kept

    if not files:
        raise RuntimeError("沒有可用的 episode")

    # 驗證第一個 file 格式
    with h5py.File(files[0], "r") as f:
        fmt = f.attrs.get("action_format", "<missing>")
        fps = int(f.attrs.get("fps", 0))
        state_dim = int(f.attrs.get("state_dim", -1))
        print(f"  格式檢查：action_format={fmt}, fps={fps}, state_dim={state_dim}")
        assert fmt == "lerobot_folding_16d", f"不是 v2 格式（{fmt}）"
        assert state_dim == 16
        assert fps == 30
    return files


# ── motion-window trim（解 SmolVLA 閉環卡在開頭 home 停留 + 砍尾端空錄）─────────
# arm joints（16D folding 去掉 gripper idx 7,15）|action|>THRESH 才算「在動」。
TRIM_LEAD   = 10     # 第一個動作前保留幾幀（lead-in，讓 policy 看到「即將動」）
TRIM_TRAIL  = 15     # 最後動作(含放完回 home)後保留幾幀（home settle；切掉集間長等待）
MOTION_THRESH = 0.1
_ARM_IDX = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]


def motion_window(actions: np.ndarray, lead: int = TRIM_LEAD, trail: int = TRIM_TRAIL) -> tuple[int, int]:
    """回傳 (start, end)：只保留有手臂動作的區段 + 頭尾少量 margin。

    lead 控頭端（解 home attractor：trimmed 開頭的 home lead-in 會讓 policy 學「維持 home」）。
    尾端保留到最後一個手臂動作(放完籃→回 home)+trail，含回 home settle；自動切掉收集腳本
    在回 home 後「等下一筆」的超長靜止（那段在 last motion 之後，不會被計入）。
    """
    n = len(actions)
    mv = np.abs(actions[:, _ARM_IDX]).max(axis=1)
    moving = np.where(mv > MOTION_THRESH)[0]
    if len(moving) == 0:
        return 0, n
    s = max(0, int(moving[0]) - lead)
    e = min(n, int(moving[-1]) + trail)
    return s, e


def build_lerobot_dataset(
    hdf5_files: list[Path],
    dataset_path: Path,
    repo_id: str,
    task_language: str,
    trim_motion: bool = False,
    trim_lead: int = TRIM_LEAD,
    trim_trail: int = TRIM_TRAIL,
):
    if dataset_path.exists():
        print(f"刪除舊的 dataset: {dataset_path}")
        shutil.rmtree(dataset_path)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (16,),
            "names": FOLDING_16D_NAMES,
        },
        "observation.images.main": {
            "dtype": "video", "shape": (3, 240, 320),
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist_left": {
            "dtype": "video", "shape": (3, 240, 320),
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist_right": {
            "dtype": "video", "shape": (3, 240, 320),
            "names": ["channels", "height", "width"],
        },
        "action": {
            "dtype": "float32",
            "shape": (16,),
            "names": FOLDING_16D_NAMES,
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        features=features,
        root=dataset_path,
        robot_type="openarm_bimanual",
        use_videos=True,
        image_writer_threads=4,
    )

    task_counts: dict[str, int] = {}
    for ep_idx, hdf5_path in enumerate(tqdm(hdf5_files, desc="轉換 episodes")):
        with h5py.File(hdf5_path, "r") as f:
            n = int(f.attrs.get("num_frames", f["obs/joint_pos"].shape[0]))
            joint_pos = f["obs/joint_pos"][:n]
            actions   = f["actions"][:n]
            cam_main  = f["obs/cam_main"][:n]
            cam_wl    = f["obs/cam_wrist_left"][:n]
            cam_wr    = f["obs/cam_wrist_right"][:n]
            # 多任務：每集用 HDF5 自帶的 task 標籤（collect_handover_demos 已寫入 attrs["task"]）；
            # 沒有才退回 --task_language。三水果各自 task → 一個多任務 dataset。
            ep_task = f.attrs.get("task", task_language)
            if isinstance(ep_task, bytes):
                ep_task = ep_task.decode()

        if trim_motion:
            s, e = motion_window(actions, trim_lead, trim_trail)
            joint_pos, actions = joint_pos[s:e], actions[s:e]
            cam_main, cam_wl, cam_wr = cam_main[s:e], cam_wl[s:e], cam_wr[s:e]
            n = len(joint_pos)
        task_counts[ep_task] = task_counts.get(ep_task, 0) + 1

        for t in range(n):
            dataset.add_frame({
                "observation.state": joint_pos[t].astype(np.float32),
                "observation.images.main": cam_main[t].transpose(2, 0, 1),
                "observation.images.wrist_left": cam_wl[t].transpose(2, 0, 1),
                "observation.images.wrist_right": cam_wr[t].transpose(2, 0, 1),
                "action": actions[t].astype(np.float32),
                "task": ep_task,
            })
        dataset.save_episode()

    print("\n各 task episode 數：")
    for tk, c in sorted(task_counts.items()):
        print(f"  [{c:3d}] {tk}")

    dataset.finalize()
    print(f"\n轉換完成！共 {dataset.num_episodes} episodes，{dataset.num_frames} frames")
    return dataset


def patch_stats(dataset_path: Path):
    """擴張右臂 zero-variance dims 的 stats，避免訓練時 normalize 除以微小值爆炸。

    跟 scripts/tools/patch_basket_v2_stats.py 同邏輯，直接合進來避免兩個地方維護。
    """
    import json
    stats_path = dataset_path / "meta" / "stats.json"
    if not stats_path.exists():
        print(f"[WARN] {stats_path} 不存在，跳過 patch")
        return

    # backup
    bak = stats_path.with_suffix(".json.bak")
    if not bak.exists():
        shutil.copy(stats_path, bak)
        print(f"  [backup] {bak}")

    stats = json.loads(stats_path.read_text())

    total_patched = 0
    for feat in ("observation.state", "action"):
        if feat not in stats:
            continue
        s = stats[feat]
        q01 = np.asarray(s["q01"], dtype=np.float64)
        q99 = np.asarray(s["q99"], dtype=np.float64)
        diff = q99 - q01
        small = np.abs(diff) < MIN_NORM_RANGE
        n_small = int(small.sum())
        if n_small == 0:
            continue

        half = MIN_NORM_RANGE / 2
        mid = (q01 + q99) / 2
        s["q01"] = np.where(small, mid - half, q01).tolist()
        s["q99"] = np.where(small, mid + half, q99).tolist()

        if "min" in s and "max" in s:
            mn = np.asarray(s["min"], dtype=np.float64)
            mx = np.asarray(s["max"], dtype=np.float64)
            s["min"] = np.where(small, mid - half, mn).tolist()
            s["max"] = np.where(small, mid + half, mx).tolist()

        if "std" in s:
            st = np.asarray(s["std"], dtype=np.float64)
            s["std"] = np.where(small, MIN_NORM_RANGE / 2, st).tolist()

        idx = np.where(small)[0].tolist()
        print(f"  [{feat}] 擴張 {n_small} 維 zero-variance dims (idx={idx})")
        total_patched += n_small

    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"  Stats patched: 共 {total_patched} 維修正，寫回 {stats_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_dir", required=True, nargs="+",
                        help="一個或多個 HDF5 episode 目錄（會 glob demo_*.hdf5）")
    parser.add_argument("--dataset_name", required=True,
                        help="輸出 dataset 名稱，例如 basket_task_fixed_grid")
    parser.add_argument("--output_root", type=str,
                        default="/home/teddyminelab/openarm_isaaclab/datasets_lerobot/openarm",
                        help="LeRobotDataset 輸出根目錄")
    parser.add_argument("--task_language", type=str,
                        default="Pick up the cube and place it into the basket.",
                        help="task description（必須跟 inference bridge TASK_DESC 完全一致）")
    parser.add_argument("--include_all", action="store_true",
                        help="不過濾 success=True，全部 episode 都納入")
    parser.add_argument("--trim_motion", action="store_true",
                        help="砍掉每集開頭/尾端的 home 停留（只留動作區段+少量margin）。"
                             "★ handover 必開：解 SmolVLA 閉環卡在開頭停留 + 省尾端空錄")
    parser.add_argument("--trim_lead", type=int, default=TRIM_LEAD,
                        help=f"第一個動作前保留幾幀。★ 設 0 解閉環 home attractor（lead-in 會讓 policy 學"
                             f"「在 home 維持 home」，訓越久越卡死）。default={TRIM_LEAD}")
    parser.add_argument("--trim_trail", type=int, default=TRIM_TRAIL,
                        help=f"最後動作後保留幾幀。default={TRIM_TRAIL}")
    args = parser.parse_args()

    hdf5_dirs = [Path(d) for d in args.hdf5_dir]
    for d in hdf5_dirs:
        if not d.exists():
            raise FileNotFoundError(f"HDF5 dir not found: {d}")

    dataset_path = Path(args.output_root) / args.dataset_name

    print(f"[1/3] Collecting HDF5 episodes from: {[str(d) for d in hdf5_dirs]}")
    hdf5_files = collect_hdf5_files(hdf5_dirs, only_success=not args.include_all)

    print(f"\n[2/3] Building LeRobotDataset at: {dataset_path}  "
          f"(trim_motion={args.trim_motion}, lead={args.trim_lead}, trail={args.trim_trail})")
    build_lerobot_dataset(hdf5_files, dataset_path, args.dataset_name, args.task_language,
                          trim_motion=args.trim_motion,
                          trim_lead=args.trim_lead, trim_trail=args.trim_trail)

    print(f"\n[3/3] Patching zero-variance stats:")
    patch_stats(dataset_path)

    print(f"\n[ALL DONE] Dataset ready: {dataset_path}")
    print(f"  Use in notebooks by setting DATASET_REPO_ID = '{args.dataset_name}'")


if __name__ == "__main__":
    main()
