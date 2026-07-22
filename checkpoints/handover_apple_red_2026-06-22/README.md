# Checkpoint: 雙臂 handover apple → basket_red（穩定版）

**日期：** 2026-06-22
**狀態：** apple → basket_red 單一流程 **100% 可靠**
**正式 benchmark：20/20 成功（100%）**，20 集 × bounds abort=0、step15 -26=0、hard_fail=0、全 17/17（results_handover_bench/2026-06-22_04-38-45/）。小批驗證另有 2/2、5/5、10/10。

## 這個 checkpoint 是什麼
左臂抓 apple → 中線正交雙抓交接給右臂 → 右臂放進紅籃 → 雙臂回 home。
是 Phase 2「指定水果→指定籃子」的第一個完整可靠組合。

## 備份的關鍵檔案
| 檔案 | 用途 |
|------|------|
| `bimanual_handover.py` | 控制端 17 步序列（→ `scripts/ros2/`）|
| `run_handover_benchmark.py` | Isaac 端 benchmark：載場景+bridge、逐集 reset、錄關節+apple pose（→ `scripts/isaac/`）|
| `run_handover_episodes.py` | ROS2 watcher：逐集重啟 move_group、跑一次 handover（→ `scripts/ros2/`）|
| `joint_limits.yaml` | **bounds 根治**：全 joint 限位 +0.02 rad（→ `ros2_ws/src/openarm_description/config/arm/v10/` 與 install/share 兩份）|
| `demo.launch.py` | move_group launch（→ `ros2_ws/.../openarm_bimanual_moveit_config/launch/`）|

## 三個關鍵修法（這版之所以 100%）
1. **bounds -26 根治** — handover 姿態把 joint3/6/7 頂到限位，物理/浮點過衝 ~0.0001–0.0003 rad，
   `move_group` 的 `CheckStartStateBounds` 直接 abort 規劃（長期看到的「-26 / 手卡住」根因）。
   move_group 參數與 Python 端夾 start_state 都無效；**治本=放寬 URDF 所有 joint 限位 +0.02 rad**。
2. **fix A：step8 大風車消除** — 左手到交接點原本用 `move_pose(q=None)`（方向全放開）+ RRTConnect
   會甩一大圈肘+腕。改為**約束方向到目前的舉起姿態 `left_grasp_q`** + 寬鬆 ori_tol，再 cartesian_rotate
   轉平。含 fallback 回 position-only 保證不退步。
3. **step14 左手退回降高** — 退回 home 的中繼點 z 由 0.40 降到 0.30，不再「舉很高」。

## 執行方式（自動化 benchmark）
```bash
# 一鍵編排（清理→起 Isaac→逐集跑→收尾）；N=集數、FRUIT=apple|strawberry
bash /tmp/run_bench.sh 20 apple
bash /tmp/run_bench.sh 20 strawberry
# 結果在 results_handover_bench/<timestamp>/summary.csv + run.hdf5
```
`--fruit` 已串通 handover/watcher/benchmark；benchmark 追蹤選定水果判入籃。手動 3 終端見 memory `project_phase2_handover_working`。

## 🍓 strawberry → basket_red（2026-06-23，19/20 = 95%）
草莓在 x=0.3146（左手 top-down reach 邊界）、矮(2.5cm)。難點是 reach + 小目標夾取容差。
- **reach 破解 = 前傾夾取** `grasp_tilt=-0.65`（approach 往 +x 前傾 ~37°，只用於 step4-7，step8 轉回 top-down→交接幾何不變）。
- **TCP 對中心 = `gx_comp=+0.013`**（前傾擴大可達區後，此補償才規劃得動）。捕捉窗口約 x∈[0.297,0.321]，調到 TCP 分布 [0.308,0.318] 置中。
- **`grasp_zoff=-0.018`**（比 apple −0.030 淺，草莓矮避免頂桌面）。
- 調校軌跡：tilt−0.45(4/10)→−0.65(4/5)→+comp0.020 過頭(4/5)→**+comp0.013 (10/10, 20集19/20)**。
- 唯一失敗=搬運中途滑落（非夾取 miss）。參數都在 STRAWBERRY dict。

## 待擴充（下一步）
- banana（細長，抓取/交接幾何需重設）
- strawberry/apple → yellow 籃（只換 `--basket yellow`，座標已參數化）
- 追草莓最後 5%（搬運滑落）：夾更緊 / 交接時機
