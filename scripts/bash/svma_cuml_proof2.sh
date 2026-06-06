#!/usr/bin/env bash
# Implementation-level proof that the cuML SvmA process is alive and using GPU.
# Usage: bash scripts/bash/svma_cuml_proof2.sh [PID]
set -u
PID="${1:-}"
if [ -z "$PID" ]; then
    PID=$(pgrep -f "scripts/python/train/train.py" | head -1)
fi
echo "Target PID = $PID"
if [ ! -d "/proc/$PID" ]; then
    echo "PID $PID not found in /proc" >&2
    exit 1
fi

snap() {
    local label="$1"
    local utime stime state threads vcs nvcs wchan
    read _ _ state _ _ _ _ _ _ _ _ _ _ utime stime _ <<<"$(cat /proc/$PID/stat)"
    threads=$(grep '^Threads:' /proc/$PID/status | awk '{print $2}')
    vcs=$(grep voluntary_ctxt_switches /proc/$PID/status | awk '{print $2}')
    nvcs=$(grep nonvoluntary_ctxt_switches /proc/$PID/status | awk '{print $2}')
    wchan=$(cat /proc/$PID/wchan)
    echo "--- ${label} at $(date '+%H:%M:%S.%N') ---"
    echo "  state=${state}  threads=${threads}  wchan='${wchan}'"
    echo "  utime=${utime} stime=${stime}"
    echo "  voluntary_ctxt_switches=${vcs}"
    echo "  nonvoluntary_ctxt_switches=${nvcs}"
}

echo "===== A. counter snapshots ====="
snap "T0"
sleep 6
snap "T1 (+6s)"
echo ""

echo "===== B. cuML / CUDA libraries loaded in PID ${PID} ====="
grep -E "libcuml|libcudart|libcublas[^.]|libcusolver|libcusparse|libraft|librmm" /proc/$PID/maps \
    | awk '{print $NF}' | sort -u
echo ""

echo "===== C. nvidia-smi dmon — 6 samples (SM%, MEM%) ====="
nvidia-smi dmon -c 6 -s u
