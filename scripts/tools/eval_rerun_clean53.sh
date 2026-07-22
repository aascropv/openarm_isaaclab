#!/bin/bash
# 補跑 clean53 10k/15k：每次之間徹底清理(GPU procs + ros2 daemon + /dev/shm/fastrtps + 較長 settle)，
# 解 sweep 中「第一個 eval 後 Isaac 沒釋放→後續 scene 起不來」的系統性 teardown 問題。
PROJ=/home/teddyminelab/openarm_isaaclab; cd "$PROJ" || exit 1
cleanup() {
  pkill -9 -f 'smolvla_rtc_bridge_node' 2>/dev/null
  pkill -9 -f 'run_handover_eval_scen' 2>/dev/null
  pkill -9 -f 'isaaclab.sh' 2>/dev/null
  pkill -9 -f 'benchmark_handover' 2>/dev/null
  pkill -9 -f 'kit' 2>/dev/null
  source /opt/ros/jazzy/setup.bash 2>/dev/null; ros2 daemon stop 2>/dev/null
  rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* 2>/dev/null
  rm -f /tmp/openarm_episode.json 2>/dev/null
  sleep 15
}
cleanup
for CKPT in 010000 015000; do
  echo "===== RERUN $CKPT ====="
  cleanup
  bash scripts/tools/eval_fruit.sh apple apple_clean53_cont "$CKPT" 20 25 8 2>&1
  RES=$(ls -t results_handover_eval/handover_apple_red_apple_clean53_cont_${CKPT}_*.csv 2>/dev/null | head -1)
  if [ -n "$RES" ]; then
    python3 -c "import csv;r=list(csv.DictReader(open('$RES')));ok=sum(str(x['success']).lower()=='true' for x in r);hm=sum(str(x['arm_returned_home']).lower()=='true' for x in r);both=sum(str(x['success']).lower()=='true' and str(x['arm_returned_home']).lower()=='true' for x in r);print('ckpt=$CKPT success=%d/%d arm_home=%d/%d both=%d/%d'%(ok,len(r),hm,len(r),both,len(r)))" | tee -a /tmp/eval_sweep_clean53_summary.txt
  else
    echo "ckpt=$CKPT STILL NO RESULT" | tee -a /tmp/eval_sweep_clean53_summary.txt
  fi
done
echo "===== RERUN DONE ====="; cat /tmp/eval_sweep_clean53_summary.txt
