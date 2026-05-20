#!/bin/bash
SCRIPT=/mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison/scripts/bash/repair_lstm_swsmote_within.sh
chmod +x $SCRIPT
nohup bash $SCRIPT >/home/ynakagama/repair_lstm.log 2>&1 &
echo $! > /home/ynakagama/repair_lstm.pid
echo "Launched PID=$(cat /home/ynakagama/repair_lstm.pid)"
sleep 2
echo "--- Log start ---"
head -8 /home/ynakagama/repair_lstm.log
