#!/usr/bin/env bash
# Periodic GPU stats logger — writes one CSV line every INTERVAL seconds.
# Usage:
#   nohup bash scripts/bash/gpu_logger.sh > /home/ynakagama/gpu_log.csv 2>&1 &
# View live:
#   tail -f /home/ynakagama/gpu_log.csv
set -u
INTERVAL="${GPU_LOG_INTERVAL:-5}"

# CSV header
echo "timestamp,gpu_util_pct,mem_used_mib,mem_total_mib,power_w,temp_c,clock_mhz,fan_pct"

while true; do
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    row="$(nvidia-smi \
        --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,clocks.current.graphics,fan.speed \
        --format=csv,noheader,nounits | tr -d ' ')"
    echo "${ts},${row}"
    sleep "${INTERVAL}"
done
