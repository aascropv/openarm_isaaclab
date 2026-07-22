#!/bin/bash
# clean53 continued-finetune 的 checkpoint sweep：依序 eval 5k/10k/15k（n=20, EH=25, hw=8，對齊 apple_v2 40k=80% baseline）。
# 每個 ckpt 各自起 Isaac→bridge→benchmark 20 集→拆場景，最後印各 ckpt 成功率彙總。
PROJ=/home/teddyminelab/openarm_isaaclab
cd "$PROJ" || exit 1
DATASET=apple_clean53_cont
N=20; EH=25; HW=8
SUMMARY=/tmp/eval_sweep_clean53_summary.txt
: > "$SUMMARY"
for CKPT in 005000 010000 015000; do
  echo "========== SWEEP ckpt=$CKPT =========="
  bash scripts/tools/eval_fruit.sh apple "$DATASET" "$CKPT" "$N" "$EH" "$HW" 2>&1
  # 抓最新該 ckpt 的 benchmark result csv，數 success
  RES=$(ls -t results_handover_eval/handover_apple_red_${DATASET}_${CKPT}_*.csv 2>/dev/null | head -1)
  if [ -n "$RES" ]; then
    OK=$(awk -F, 'NR>1 && tolower($0) ~ /,true,|,true$/{c++} END{print c+0}' "$RES")
    # 更穩：用 success 欄
    OK=$(python3 -c "import csv,sys;r=list(csv.DictReader(open('$RES')));print(sum(1 for x in r if str(x.get('success','')).strip().lower()=='true'))" 2>/dev/null)
    HOME=$(python3 -c "import csv;r=list(csv.DictReader(open('$RES')));print(sum(1 for x in r if str(x.get('arm_returned_home','')).strip().lower()=='true'))" 2>/dev/null)
    TOT=$(python3 -c "import csv;print(len(list(csv.DictReader(open('$RES')))))" 2>/dev/null)
    echo "ckpt=$CKPT  success=$OK/$TOT  arm_home=$HOME/$TOT  ($RES)" | tee -a "$SUMMARY"
  else
    echo "ckpt=$CKPT  NO RESULT CSV" | tee -a "$SUMMARY"
  fi
  sleep 5
done
echo "===== SWEEP DONE ====="; cat "$SUMMARY"
