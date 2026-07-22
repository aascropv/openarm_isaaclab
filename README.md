# OpenArm 雙臂手遞手（Bimanual Handover）— Isaac Sim + LeRobot

雙臂機械手臂在 Isaac Sim 中完成「**右手抓水果 → 空中手遞手交接 → 左手放進籃子 → 雙臂歸位**」的模仿學習專案。

- **三個任務**：`apple → 紅籃`、`strawberry → 紅籃`、`banana → 黃籃`
- **現行策略**：**ACT**（Action Chunking Transformer），**每個水果各訓一個獨立模型**
- **技術棧**：Isaac Sim / Isaac Lab（模擬 + 資料收集）· ROS2 Jazzy（控制與 policy bridge）· LeRobot（資料格式與訓練）

---

## ⚠️ 命名陷阱（最重要，先看這個）

「**v2**」在這個專案裡**指過兩個不同世代的資料**。看到 v2 一定要確認是哪一個：

| 世代 | 目錄 | 收集日期 | 現況 |
|---|---|---|---|
| 1 | ~~`datasets_handover/OLD_v1_pre-returnpath-tuning`~~ | 06-28~07-01 | 已刪除 |
| 2 | ~~`datasets_handover_100/`~~ | 07-09 | 已刪除。**它的 README 自稱「v2」**，但本專案後續文件一律稱它 **v1** |
| 3 | **`datasets_handover_v2/`** | **07-17** | ⭐ **現行資料來源** |

同理，`datasets_lerobot/openarm/` 底下：`handover_apple`＝**v1**、`handover_apple_v2`＝**v2（現行）**。

---

## 現行成果

三果各自獨立模型，各用**自己最佳的 checkpoint**（合併 sweep + 錄影共 N=20 實測）：

| 水果 | 最佳 ckpt | 入籃成功率 | 入籃＋回 home | 對照 v1（40k） |
|---|---|---|---|---|
| **apple**（紅籃） | **100k** | **100%** | **85%** | 70% / 50% |
| **strawberry**（紅籃） | **80k** | **100%** | **90%** | 100% / 90% |
| **banana**（黃籃） | **60k** | **85%** | **80%** | 100% / **10%** |

> 最大成果：**banana 的「回 home」從 v1 的 10% 修到 80%**。v2 靠的是重收乾淨資料（新相機 + grasp 修正 + 甩圈/甩飛篩選）與轉檔時的頭尾 trim。

**完整曲線（N=20，入籃 / 入籃＋回 home）**

| ckpt | apple | strawberry | banana |
|---|---|---|---|
| 60k | 95 / 65 | 55 / 30 | **85 / 80** |
| 70k | 80 / 80 | 95 / 75 | 75 / 55 |
| 80k | 90 / 85 | **100 / 90** | 75 / 45 |
| 90k | 95 / 80 | 85 / 70 | 65 / 55 |
| 100k | **100 / 85** | 100 / 65 | 80 / 65 |

**三個關鍵觀察**

1. **甜蜜點大幅後移**：v1 三果都在 40k，v2 全在 60–100k。trim 後資料更密實乾淨 → 需要更多步收斂（v1 的 40k 其實是欠訓）。
2. **訓練 loss 單調下降到 100k，但成功率不是**。loss 只量「多會背 demo」，不量閉環韌性 → **必須逐 ckpt 實測，不能只看 loss 或直接用最後一版**。
3. **回 home 最先崩**：它在每集尾端（誤差累積最重），且 trim 後只留 15 幀（嚴重欠代表）→ 過擬合會最先弄壞它。strawberry 100k 就是入籃仍 100%、回 home 掉到 65%。

---

## 目錄結構

```
openarm_isaaclab/
├── datasets_handover_v2/          78G  ⭐ 現行原始 HDF5（三果各 100 乾淨 + excluded/ spare/）
│   └── <fruit>/<fruit>_<時間戳>/demo_*.hdf5
├── datasets_lerobot/openarm/      12G  訓練用 LeRobotDataset
│   ├── handover_{fruit}_v2             ⭐ 現行（trim lead12/trail15）
│   ├── handover_{fruit}                v1（保留供對照重驗）
│   └── apple_*/strawberry_v1/...       SmolVLA 時期的實驗資料集
├── outputs/                       80G  模型與 log
│   ├── finetune_act_handover_{fruit}_v2/   ⭐ 現行 ACT（10 個 ckpt：10k–100k）
│   ├── finetune_act_handover_{fruit}/      v1 ACT（只留 40k，供現場對照展示）
│   ├── finetune_act_basket/                ACT 配方範本（單任務 basket 90%）
│   ├── finetune_smolvla_*/                 SmolVLA 系列（保留）
│   └── evalview_act_handover_*_v2_*        15 支評測錄影（3cam mp4 + EEF csv）
├── results_handover_eval/        424K  ⭐ 所有評測的逐集原始 CSV（v1+v2 數字的佐證）
├── scripts/
│   ├── tools/                          收集 / 篩選 / 轉檔 / 訓練 / 評測 / 錄影 / 檢視
│   ├── isaac/                          Isaac Sim 場景與收集端
│   └── ros2/                           ROS2 控制端（含 bimanual_handover.py 動作核心）
├── policy_bridges/                     policy → ROS2 推論橋接（ACT / SmolVLA）
├── notebooks/{act,smolvla}/            訓練監控與實驗 notebook
├── assets/                       515M  USD 場景（openarm_bimanual_env.usd）
├── checkpoints/                  1.3M  里程碑程式碼備份（附 README，非模型權重）
└── ros2_ws/                      386M  ROS2 workspace
```

---

## 環境

| conda env | 用途 |
|---|---|
| `env_isaaclab` | Isaac Lab / Isaac Sim、USD 編輯（`pxr`）、資料收集與評測場景 |
| `lerobot` | 訓練、資料轉檔、policy bridge、所有 GUI viewer |
| `env_isaacsim` | Isaac Sim 獨立用途 |

> **`conda run` 一定要加 `--no-capture-output`**，否則 stdout 會被完整緩衝，看不到即時進度。

---

## 完整 Pipeline

### 1. 收集資料（Isaac Sim，會開閉環模擬）

```bash
# 單果收到滿 100 筆「乾淨」demo（自動迴圈：收集 → 篩選 → 補收 → cap）
bash scripts/tools/collect_clean_fruit.sh apple red 100

# 三果依序
bash scripts/tools/collect_clean_all.sh 100
```

篩選有 **6 個標準**（`filter_fruit_demos.py`）：任務成功、落在目標籃 bbox、起步在 home、末幀回 home、無甩飛（水果 z < 0.6）、無甩圈（右臂 base 轉幅 < 2.5 rad）。不合格搬到 `excluded/`，超額搬到 `spare/`。

### 2. 轉成 LeRobotDataset（含頭尾 trim）

```bash
bash scripts/tools/convert_v2_all.sh        # 三果，trim_motion lead12/trail15
```

**為什麼一定要 trim**：原始每集約 1100 幀，其中**頭約 90 幀（3 秒）、尾約 350 幀（11 秒）是手臂完全靜止的廢幀**（近 40%）。不砍掉會讓 policy 學成「傾向不動」，開頭還會卡在 home 不肯動（home attractor）。trim 後約 650 幀/集。

### 3. 訓練 ACT

```bash
bash scripts/tools/train_v2_all.sh          # 三果依序 100k，存每 10k（約 52 分鐘/果）
bash scripts/tools/train_act_fruit.sh handover_apple_v2 100000   # 單果
```

**配方**（照抄已跑通的 basket ACT，並與官方 ACT 文件一致）：
`chunk_size=60 / n_action_steps=1 / temporal_ensemble_coeff=0.01 / kl_weight=10 / batch 8 / lr 1e-5 / seed 42`

> ⚠️ **刻意偏離 lerobot 預設**：預設是 `chunk=100 / n_action_steps=100 / 無 temporal ensembling`（開環）。本專案用的是**原始 ACT 論文的閉環 temporal ensembling 模式**（每步重新推論 + 指數加權融合），對接觸密集的雙臂操作更好。其餘超參數（lr、優化器、ResNet18 backbone、kl_weight、CVAE）全用官方預設。

訓練監控：`notebooks/act/monitor_act_train.ipynb`（改 `LOG_PATH` / `OUTPUT_DIR` 即可切換水果）。

### 4. 評測 sweep

```bash
bash scripts/tools/sweep_v2_all.sh 10 040000 050000 060000 070000 080000 090000 100000
bash scripts/tools/eval_act_fruit.sh apple handover_apple_v2 100000 10   # 單 ckpt
```

結果寫進 `results_handover_eval/*.csv`（逐集含 `success`、`arm_returned_home`、失敗原因、水果起訖座標）。

### 5. 錄影 + 檢視

```bash
bash scripts/tools/record_v2_menu.sh 10 060000 070000 080000 090000 100000   # 錄 3cam mp4 + EEF csv
conda run --no-capture-output -n lerobot python3 scripts/tools/eval_menu_viewer.py
```

`eval_menu_viewer.py` 是**選單式 GUI**：下拉選 **水果 → ckpt → episode**（episode 直接顯示 `in=Y/N home=Y/N`），畫面同時呈現三相機、16 關節、雙臂側視（看有無撞平台）。錄影陸續產生時按 **🔄 rescan** 更新選單。

檢視**訓練資料**（非評測）用：
```bash
conda run --no-capture-output -n lerobot python3 scripts/tools/dataset_viewer.py --fruit apple
```

---

## 腳本用途對照表

> 腳本之間有 **51 處寫死的相對路徑互相呼叫**（例如 `sweep_v2_all.sh → eval_sweep_act.sh → eval_act_fruit.sh → run_handover_eval_scene.sh`）。**不要搬動檔案位置**，否則要同步修改所有引用。

### 收集
| 腳本 | 用途 |
|---|---|
| `tools/collect_clean_all.sh` | 三果依序各收滿 N 筆乾淨 demo |
| `tools/collect_clean_fruit.sh` | 單果迴圈：收集 → 篩選 → 補收 → cap（可中斷續收） |
| `tools/collect_fruit.sh` | 通用單果收集（底層） |
| `isaac/collect_handover_demos.py` | Isaac 端收集主程式 |
| `ros2/bimanual_handover.py` | ⭐ **動作核心**：17 步 handover 序列、三果各自 grasp 參數、反向重播回程 |

### 篩選與轉檔
| 腳本 | 用途 |
|---|---|
| `tools/filter_fruit_demos.py` | 6 標準事後篩選（含甩圈/甩飛偵測） |
| `tools/hdf5_to_lerobot_dataset.py` | HDF5 → LeRobotDataset，含 `--trim_motion` 與 stats zero-variance patch |
| `tools/convert_v2_all.sh` | 三果一次轉完（trim 12/15） |

### 訓練
| 腳本 | 用途 |
|---|---|
| `tools/train_v2_all.sh` | ⭐ 三果 ACT 依序 100k |
| `tools/train_act_fruit.sh` | 單果 ACT 訓練 |
| `tools/train_fruit.sh` / `train_apple.sh` / `train_apple_clean53_cont.sh` | SmolVLA 時期訓練腳本 |

### 評測
| 腳本 | 用途 |
|---|---|
| `tools/sweep_v2_all.sh` | ⭐ 三果 × 多 ckpt sweep |
| `tools/eval_sweep_act.sh` | 單果多 ckpt sweep（ckpt 間 strong cleanup） |
| `tools/eval_act_fruit.sh` | ⭐ 單 ckpt ACT 閉環評測 |
| `tools/benchmark_handover.py` | 評測 orchestrator（判定入籃與回 home） |
| `tools/benchmark_inference.py` | 通用多集 benchmark orchestrator（ACT / SmolVLA 皆可） |
| `tools/eval_fruit.sh` / `eval_sweep_v2.sh` / `eval_sweep_clean53.sh` / `eval_rerun_clean53.sh` | SmolVLA 時期評測 |
| `tools/eval_banana_retest.sh` | banana 乾淨重測（隔離 Isaac degrade 污染） |

### 錄影與檢視
| 腳本 | 用途 |
|---|---|
| `tools/record_v2_menu.sh` | ⭐ 批次錄影供選單 viewer |
| `tools/eval_record_view_act.sh` | 單組 ACT 錄影（3cam mp4 + EEF csv） |
| `tools/eval_recordview_all_act.sh` | 三果 ACT 批次錄製複測（逐果跑上者、果間 strong cleanup） |
| `tools/eval_record_view.sh` | SmolVLA 版錄影（ACT 版的前身） |
| `tools/visualize_dataset.py` | Dataset 視覺化工具 |
| `tools/eval_menu_viewer.py` | ⭐ 選單式評測檢視 GUI |
| `tools/eval_viewer.py` | 單組評測逐幀檢視（需 `--base`） |
| `tools/dataset_viewer.py` | ⭐ 訓練資料（HDF5）逐幀檢視 |
| `tools/record_3cam.py` / `record_camera.py` / `eval_record_one.sh` | 底層錄影工具 |
| `tools/demo_to_video.py` / `replay_handover_demo.py` | HDF5 → mp4 |

### Isaac 場景
| 腳本 | 用途 |
|---|---|
| `isaac/run_handover_eval_scene.sh/.py` | ⭐ 評測場景（3 水果 2 籃固定位置 + episode reset） |
| `isaac/run_handover_benchmark.py` | 可靠度 benchmark |
| `isaac/run_inference_scene.sh` | 推論場景 |
| `isaac/run_benchmark_scene.sh` | benchmark 場景（含 cube 隨機化 + episode reset） |
| `ros2/launch_ros2_stack.sh` | 啟動 Isaac Lab 整合所需的 ROS2 stack |

### Policy Bridge
| 檔案 | 用途 |
|---|---|
| `policy_bridges/act_bridge_node.py` + `run_act_bridge.sh` | ⭐ ACT 推論橋接（ready marker：`ACT bridge ready`） |
| `policy_bridges/bridge_common.py` | 共用常數（16D ↔ 18D mimic 轉換等） |
| `policy_bridges/smolvla_rtc_bridge*` | SmolVLA + RTC 橋接（多版本） |

### 診斷
`tools/plot_eef_trajectory.py`、`record_eef_trajectory.py`、`log_joint_trajectory.py`、`run_log_joint_trajectory.sh`、`log_gripper_diag.py`、`debug_articulation.py`、`add_camera_ros2_publishers.py`（在 Isaac Sim Script Editor 內執行，掛相機 ROS2 publisher）、`isaac/query_banana.py`

---

## 踩雷筆記

**流程類**

- **`pkill -f <pattern>` 會自殺**：如果執行中的指令列本身含有該 pattern，`pkill` 會掃到自己那個 shell。**清理一定要寫在獨立的 script 檔裡**（`bash xxx.sh` 的 cmdline 不含 pattern），不要直接在指令列打。
- **收集/評測期間不要另外開 Isaac Sim 檢視** → 兩個 Isaac 搶 GPU → 收集端 hang（GPU util 掉 0%）。要看資料請用 `dataset_viewer.py`（不吃 GPU）。
- 背景發車那行 Bash 常回 **exit 1**（`pgrep | grep -v grep` 的退出碼），那是紅鯡魚 —— 用 `pgrep` 確認程序真的起來了再說，不要重複發車。

**實驗方法類**

- **N=10 的評測雜訊很大**：同一個 ckpt 兩次獨立 N=10 可以差 30–40%（banana 尤其嚴重）。**結論要用合併後的 N=20 以上**，別憑單次結果下判斷。
- **不要只用最後一個 ckpt**，也不要只看 loss（見上方「三個關鍵觀察」）。每個水果的最佳 ckpt 不同。
- lerobot v0.5 的坑：訓練 SmolVLA 時用 `--policy.path` 會讓 state 維度卡在預設值，要用 `--policy.type` + `--policy.load_vlm_weights=true`。

**歷史遺跡**

- `notebooks/*/finetune_*_basket*.ipynb` 與 `outputs/finetune_act_basket`：**basket 任務的 dataset（`basket_task_v2`）已不存在**，這些 notebook 無法重跑，僅作為 ACT 配方的來源保留。
- `policy_bridges/bridge_common.py` 等檔案的註解會提到 `pi05_bridge_node.py` —— 該檔已在 PI0.5 路線放棄後刪除，註解是沿革記錄，刻意保留。
