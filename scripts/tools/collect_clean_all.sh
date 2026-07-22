#!/bin/bash
# 三果依序各收滿 TARGET 筆乾淨 demo（apple→red, strawberry→red, banana→yellow）。
# 用法: bash scripts/tools/collect_clean_all.sh [target=100]
set -u
PROJ=/home/teddyminelab/openarm_isaaclab
cd "$PROJ" || exit 1
TARGET=${1:-100}
echo "[all] ===== 三果各收 $TARGET 乾淨 start=$(date '+%m-%d %H:%M') ====="
for pair in "apple red" "strawberry red" "banana yellow"; do
  set -- $pair
  fruit=$1; basket=$2
  echo "[all] >>>>> $fruit→$basket ($(date '+%m-%d %H:%M')) <<<<<"
  bash scripts/tools/collect_clean_fruit.sh "$fruit" "$basket" "$TARGET"
done
echo "[all] ===== ALL 3 FRUITS DONE $(date '+%m-%d %H:%M') ====="
echo "[all] 結果:"
for f in apple strawberry banana; do
  n=$(find "datasets_handover_v2/$f" -mindepth 2 -maxdepth 2 -name 'demo_*.hdf5' 2>/dev/null | wc -l)
  echo "  $f: $n 乾淨"
done
