#!/bin/bash
REPO=/mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison
PYTHON=/home/ynakagama/.venv_tf_gpu/bin/python
LOGFILE=/home/ynakagama/launcher_other.log
PIDFILE=/home/ynakagama/launcher_other.pid
cd $REPO
# setsid + nohup + redirect-everything-from-/dev/null fully detaches the python
# process from the calling shell session so it survives WSL session exit.
setsid nohup $PYTHON scripts/python/train/local_exp3_lstm_wsl2_other_launcher.py \
    </dev/null >>$LOGFILE 2>&1 &
PID=$!
echo $PID > $PIDFILE
echo "Started PID=$PID"
# Give it a moment so the process is fully established before we return.
sleep 2