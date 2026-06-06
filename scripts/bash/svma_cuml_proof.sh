#!/usr/bin/env bash
# One-shot diagnostic: snapshot then re-snapshot 6 seconds later
# to prove that PID has made progress (CPU, ctxt switches, I/O).
set -u
PID="${1:-}"
if [ -z "$PID" ]; then
    PID=$(pgrep -f "train.py.*SvmA" | head -1)
fi
echo "Target PID = $PID"
if [ ! -d "/proc/$PID" ]; then
    echo "PID $PID not found." >&2; exit 1
fi

snap() {
    local label="$1"
    echo "--- ${label} at $(date +%H:%M:%S.%N) ---"
    awk '{printf "  utime=%s stime=%s\n", $14, $15}' /proc/$PID/stat
    grep -E "^(voluntary_ctxt_switches|nonvoluntary_ctxt_switches|State|Threads|VmRSS):" /proc/$PID/status \
        | sed 's/^/  /'
    grep -E "^(rchar|wchar|read_bytes|write_bytes):" /proc/$PID/io 2>/dev/null \
        | sed 's/^/  /'
    echo "  syscall: $(awk '{print $1}' /proc/$PID/syscall 2>/dev/null) (read=0 write=1 ioctl=16 poll=7 nanosleep=35)"
    echo "  wchan:   $(cat /proc/$PID/wchan 2>/dev/null) (0=running on CPU)"
}

snap "T0"
sleep 6
snap "T1 (+6s)"

echo ""
echo "===== If counters increased, the process IS executing instructions. ====="
