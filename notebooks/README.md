# notebooks — basket 任務 finetune / 監控

依 policy 分兩個子夾。**pi0.5 已棄用**（多次 finetune 失敗，改用 ACT/SmolVLA），相關 notebook 已刪除。

## `act/` — 主力（benchmark ~90%）
| 檔案 | 用途 |
|---|---|
| `finetune_act_basket.ipynb` | ACT finetune basket 任務 |

## `smolvla/` — 次要（benchmark ~85%）
| 檔案 | 用途 |
|---|---|
| `finetune_smolvla_basket.ipynb` | 初版 finetune |
| `finetune_smolvla_basket_consistent.ipynb` | 一致性資料版 |
| `finetune_smolvla_basket_consistent_v2.ipynb` | 一致性 v2（乾淨 baseline） |
| `finetune_smolvla_basket_top500.ipynb` | top-500 demo 篩選版 |
| `monitor_smolvla_train.ipynb` | 訓練監控 |

> SmolVLA 注意：訓練要用 `--policy.type=smolvla` + `--policy.load_vlm_weights=true`，不要用 `--policy.path`（會讓 state 卡在 [6]）。
