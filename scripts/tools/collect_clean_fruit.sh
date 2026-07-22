#!/bin/bash
# 收集「乾淨」demo 到滿 TARGET 筆：迴圈 收集→filter(6標準)→不夠再補收，最後 cap 到剛好 TARGET。
# filter 篩掉：非success/沒入籃/起步沒對齊/雙臂沒回home/甩飛/右臂轉圈(arm_spin)。
# 用法: bash scripts/tools/collect_clean_fruit.sh <fruit> <basket> [target=100] [max_rounds=8]
set -u
PROJ=/home/teddyminelab/openarm_isaaclab
cd "$PROJ" || exit 1
FRUIT=${1:?用法: collect_clean_fruit.sh <fruit> <basket> [target] [max_rounds]}
BASKET=${2:?缺 basket}
TARGET=${3:-100}
MAXR=${4:-8}
OUTDIR=datasets_handover_v2/$FRUIT
mkdir -p "$OUTDIR"
LOG=outputs/collect_clean_${FRUIT}.log

count_clean() { find "$OUTDIR" -mindepth 2 -maxdepth 2 -name 'demo_*.hdf5' 2>/dev/null | wc -l; }
run_filter() {
  conda run --no-capture-output -n lerobot python3 scripts/tools/filter_fruit_demos.py \
      --fruit "$FRUIT" --basket "$BASKET" --dir "$OUTDIR"/*/ "$@" 2>&1
}

echo "[clean] ===== $FRUIT→$BASKET target=$TARGET outdir=$OUTDIR start=$(date '+%m-%d %H:%M') =====" | tee -a "$LOG"
for round in $(seq 1 "$MAXR"); do
  clean=$(count_clean)
  echo "[clean] $FRUIT round $round: 目前乾淨=$clean / $TARGET  ($(date '+%H:%M'))" | tee -a "$LOG"
  [ "$clean" -ge "$TARGET" ] && { echo "[clean] 已達標" | tee -a "$LOG"; break; }
  need=$((TARGET - clean))
  batch=$(( need * 13 / 10 + 5 ))   # 補 need，×1.3 補 ~30% filter 損耗 + margin
  echo "[clean] 收 $batch 筆(補 $need 乾淨)…" | tee -a "$LOG"
  bash scripts/tools/collect_fruit.sh "$FRUIT" "$BASKET" "$batch" "$OUTDIR" 2>&1 | tail -3 | tee -a "$LOG"
  echo "[clean] filter 剔除壞的…" | tee -a "$LOG"
  run_filter --apply | tee -a "$LOG"
done
# 最後 cap 到剛好 TARGET（多的乾淨集 → spare/ 保留備用）
echo "[clean] 最終 cap 到 $TARGET…" | tee -a "$LOG"
run_filter --cap "$TARGET" --apply | tee -a "$LOG"
final=$(count_clean)
echo "[clean] ===== $FRUIT DONE: 乾淨 $final 筆 在 $OUTDIR  end=$(date '+%m-%d %H:%M') =====" | tee -a "$LOG"
