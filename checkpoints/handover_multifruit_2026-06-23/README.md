# Checkpoint: 雙臂 handover 多水果版（apple/strawberry/banana）

**日期：** 2026-06-23
**狀態：** 三種水果都跑通，水果→籃子已參數化（`FRUITS` / `BASKETS` dict）

| 水果 | 籃子 | 成功率 | 備註 |
|------|------|--------|------|
| apple | red | 20/20 = 100% | 基準（2026-06-22） |
| strawberry | red | 19/20 = 95% | reach 邊界：傾斜擴可達區 + gx_comp 補償 |
| banana | yellow | 8/10 = 80% | 長條物：右爪姿態 B + 夾質心；剩 2 集失敗在放置階段 |

接續 `handover_apple_red_2026-06-22/`（apple 100% 純淨版）。本版多了 strawberry + banana 分支。

## 備份檔案
| 檔案 | 還原到 |
|------|--------|
| `bimanual_handover.py` | `scripts/ros2/` |
| `run_handover_benchmark.py` | `scripts/isaac/`（已清掉 banana_geom debug block）|
| `run_handover_episodes.py` | `scripts/ros2/` |
| `joint_limits.yaml` | `ros2_ws/src/openarm_description/config/arm/v10/` + install/share 兩份 |
| `run_bench.sh` | `/tmp/`（編排：清理→起 Isaac→等就緒→起 watcher）|

執行：`bash run_bench.sh N {apple|strawberry|banana} {red|yellow}`

## banana 核心修法（本版重點）
長條物（138×55×33mm，交接時長軸沿世界 X 躺平）。三次嘗試才解：
1. **偏心夾持 ho_dx=0.045（錯）** → 左手放手瞬間香蕉繞夾持點重力扭翻 ~130° 脫出（HDF5 實證）。0/5。
2. **鏡像姿態 rpy(-π/2,0,0)（錯）** → 右腕 joint7 頂極限，所有 Cartesian 失敗。
3. **★ 姿態 B `right_handoff_q_banana = rpy(π/2,0,π)` + `ho_dx=0`（對）**：
   - 物理：長條物夾爪兩指必須跨「短軸」閉合，否則沿長軸並排只是戳本體表面。
   - URDF 指 prismatic `axis=0 ±1 0`→指沿 eef local-Y 閉合。原 `rpy(0,-π/2,-π/2)` 的 local-Y→world+X（=香蕉長軸）=戳。
   - B 讓 local-Y→world-Z（指上下並排跨 3.3mm 厚度，從 -y 滑入含住香蕉）；繞接近軸比鏡像 A 反向 180°→joint7 落中段、Cartesian 可達。
   - `ho_dx=0` 夾質心截面最穩，消除偏心翻轉。

## banana 已知殘留瓶頸（80%→更高需處理）
剩 2 集失敗都在**放置階段**（非交接）：香蕉跨最薄厚度單截面夾持，撐得過交接+運送，但 step15 從姿態 B 轉 top-down + 右臂舉高的重新定向把它扭滑出 → 衝過籃或飛底座旁掉地。下一步若要 90%+ 要修放置（減少重新定向 / 壓低運送舉高 / 更輕放置）。

## 其它（沿用 apple 版）
- **-26 根因解**：joint_limits.yaml 全 joint +0.02 rad（CheckStartStateBounds abort 根治）。
- fix A 大風車消除（step8 約束方向）、step14 退回降高。
- 詳見 memory `project_phase2_handover_working`。
