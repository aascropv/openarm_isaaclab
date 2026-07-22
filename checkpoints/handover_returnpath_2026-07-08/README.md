# handover 回程軌跡調順版備份（2026-07-08）

user 目視 + 軌跡圖確認的版本。主要調 handover **回程**（step14 左手 / step17 右手）motion planning，
**apple / strawberry / banana 三顆水果全部驗證通過**（兩臂 z_max≈0.37–0.43、都回 home、右手守 -y 側）。

## 檔案
- `bimanual_handover.py` — 控制端主檔（scripts/ros2/ 的副本）
- `record_eef_trajectory.py` — EEF 軌跡錄製工具（scripts/tools/ 的副本）
- `plot_eef_trajectory.py` — 軌跡畫圖工具（scripts/tools/ 的副本；已設 Noto CJK 字型，中文不變框框）
- `traj_apple.png` / `traj_strawberry.png` / `traj_banana.png` — 三顆水果各自的乾淨軌跡圖

## 這版相對前一版（handover_multifruit_2026-06-23）的改動
1. **回程 dense reverse-replay**：30Hz timer(`_sample_paths`) 在 `rec_on` 期間錄兩臂實測關節路徑到 `Arm._path`，回程 `replay_return()` 反向密集播放。取代稀疏關鍵構型內插（稀疏會在關節空間甩出巨大弧，實測 Z=1.08m）。
2. **per-arm gating**：
   - 左手錄 `home→pre-grasp` + `lift→交接點`（跳過下探夾取；接縫在 pre-grasp=lift 安全）。
   - 右手錄兩段 chunk1`home→approach` + chunk2`pull-back→放籃`，回程反向整段=`放籃→籃上→pull-back→approach→home`，**全 dense retrace 到 home、不用 move_home**（不錄 approach→insert 段，避免回程橫移回交接處中線）。
   - 註：曾試「只錄 chunk2 + move_home 收尾」，strawberry 的 move_home 從 pull-back 甩肘到 z=0.89 → 棄用。move_home 收尾不穩，pull-back→home 也要 dense。
3. 左手直退朝 home 行不通（交接近奇異，OMPL 甩肘 +y=0.5）→ 用回 reverse-replay，代價=經夾取點上方。
4. 放籃開夾爪(step16) 改 `force=True`（normal MoveGroup 偶發 -2、夾爪沒開→水果沒放掉）。
5. step8 轉水平改 `reorient_to`（OMPL reorient + TF 驗證角度，err ~8°）。
6. step2 補 `obs_<fruit>` 碰撞框（另外兩顆水果），去程 OMPL 避開；step5 下探前移除。
7. **抓取 close(step6) 改 `force=True`**：normal close 閉合去夾物體被判碰撞→ -2/99999 夾爪沒閉→空夾（strawberry 觸發）。
8. **step7 lift 改 Cartesian 直上 + `avoid_collisions=False`**：strawberry 前傾姿態閉爪被判 START_STATE_IN_COLLISION(-10)、OMPL abort；往上抬本淨空故跳過檢查安全。

## 執行（3 終端 + 背景錄製）
1. `conda run -n env_isaaclab ~/IsaacLab-2.3.0/isaaclab.sh -p scripts/isaac/run_scene_bridge.py`
2. `source ros2_ws/install/setup.bash; ros2 launch openarm_bimanual_moveit_config demo.launch.py use_isaac_sim:=true launch_rviz:=false`
3. `python3 scripts/ros2/bimanual_handover.py --fruit apple --basket red`
- （選）錄軌跡：`python3 scripts/tools/record_eef_trajectory.py --out outputs/x.csv`；畫圖 `conda run -n lerobot python3 scripts/tools/plot_eef_trajectory.py outputs/x.csv`

## 仍待處理（此版可接受、未做）
- 去程 step4 `home→pre-grasp` OMPL 繞到 +y≈0.47（reverse-replay 會 retrace，來回都繞）。
- step10 右手 approach 仍是最不穩的一步（Cartesian 有時 36%、-4 重試）。
