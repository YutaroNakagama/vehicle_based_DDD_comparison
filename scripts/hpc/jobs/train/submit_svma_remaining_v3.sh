#!/bin/bash
# Submit remaining SvmA dt array subtasks from task file v3
# Run this when queue slots become available
# Usage: bash submit_svma_remaining_v3.sh [start_index]
#   99-100: LONG 14987801, 101-113: DEFAULT 14988181
#   Default start: 114

TASK_FILE=scripts/hpc/logs/train/task_files/array_svma_domain_train_remaining_v3.txt
SCRIPT=scripts/hpc/jobs/train/pbs_array_svma.sh

REMAINING_START=${1:-114}
REMAINING_END=125

if [ "$REMAINING_START" -gt "$REMAINING_END" ]; then
    echo "All subtasks already submitted!"
    exit 0
fi

echo "Attempting to submit indices ${REMAINING_START}-${REMAINING_END} ($(( REMAINING_END - REMAINING_START + 1 )) subtasks)"
echo ""

submitted=0
idx=$REMAINING_START

for queue in DEFAULT SINGLE SMALL LONG; do
    if [ "$idx" -gt "$REMAINING_END" ]; then break; fi
    echo "--- Trying ${queue} for indices ${idx}-${REMAINING_END} ---"
    result=$(qsub -J ${idx}-${REMAINING_END} -q ${queue} -v TASK_FILE=${TASK_FILE} ${SCRIPT} 2>&1)
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "  Submitted ${idx}-${REMAINING_END} on ${queue}: $result"
        submitted=$(( submitted + REMAINING_END - idx + 1 ))
        idx=$(( REMAINING_END + 1 ))
        break
    fi
    # Try half the range
    mid=$(( (idx + REMAINING_END) / 2 ))
    if [ "$mid" -gt "$idx" ]; then
        result=$(qsub -J ${idx}-${mid} -q ${queue} -v TASK_FILE=${TASK_FILE} ${SCRIPT} 2>&1)
        rc=$?
        if [ $rc -eq 0 ]; then
            echo "  Submitted ${idx}-${mid} on ${queue}: $result"
            submitted=$(( submitted + mid - idx + 1 ))
            idx=$(( mid + 1 ))
            continue
        fi
    fi
    echo "  ${queue}: full (per-user limit)"
done

echo ""
if [ "$idx" -le "$REMAINING_END" ]; then
    echo "Submitted ${submitted} subtasks. ${idx}-${REMAINING_END} still pending ($(( REMAINING_END - idx + 1 )) left)"
    echo "Re-run later: bash $0 ${idx}"
else
    echo "All ${submitted} subtasks submitted!"
fi
