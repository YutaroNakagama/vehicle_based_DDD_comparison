#!/usr/bin/env bash
# c1 domain-comparison watchdog — drives the revised exp3 grid to completion,
# independent of any session. Within(target_only) vs Cross(source_only) x in/out
# x 4 models x 3 seeds (3 new conditions; Within-in reused from B1).
#
# CPU: RF, SvmW (Windows python). GPU: Lstm, SvmA (WSL2; one at a time, Lstm first
# then SvmA). "pending" is computed from the launcher's own --dry-run (resume-safe).
# Self-deletes the scheduled task when all 4 models report 0 pending.
set +e
REPO="/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison"
WREPO="/mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison"
cd "$REPO" || exit 1
LOGD="$REPO/logs/exp3_c1dom"; mkdir -p "$LOGD"
WLOG="$LOGD/_watchdog.log"
WINPY="/c/Users/ynakagama/AppData/Local/Programs/Python/Python311/python"
LSTMPY="/home/ynakagama/.venv_tf_gpu/bin/python"
SVMAPY="/home/ynakagama/.venv_svma_cuml/bin/python"
LAUNCH="scripts/python/train/c1_domain_launcher.py"
ts(){ date '+%Y-%m-%d %H:%M:%S'; }
pend(){ PYTHONPATH=. "$WINPY" "$REPO/$LAUNCH" --model "$1" --dry-run 2>/dev/null | grep -c imbalv3; }
alive(){ [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
gpu_busy(){ for pf in "$LOGD/.Lstm.pid" "$LOGD/.SvmA.pid"; do alive "$pf" && return 0; done; return 1; }

start_win(){ # $1=model $2=workers $3=ntrials
  cd "$REPO" || return
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 N_TRIALS_OVERRIDE="$3" PYTHONPATH=. \
    nohup "$WINPY" "$LAUNCH" --model "$1" --workers "$2" >> "$LOGD/_run_$1.log" 2>&1 &
  echo $! > "$LOGD/.$1.pid"; echo "$(ts) [c1] launched $1 (win, pid $!)" >> "$WLOG"
}
start_wsl(){ # $1=model $2=venvpy $3=workers $4=extraenv $5=limit(optional)
  local LIM=""; [ -n "$5" ] && LIM="--limit $5"
  nohup wsl -e bash -lc "cd $WREPO && CUDA_VISIBLE_DEVICES=0 $4 PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=3 $2 $LAUNCH --model $1 --workers $3 $LIM" >> "$LOGD/_run_$1.log" 2>&1 &
  echo $! > "$LOGD/.$1.pid"; echo "$(ts) [c1] launched $1 (wsl, pid $!)" >> "$WLOG"
}

PRF=$(pend RF); PSW=$(pend SvmW); PLS=$(pend Lstm); PSA=$(pend SvmA)
echo "$(ts) [c1] pending: RF=$PRF SvmW=$PSW Lstm=$PLS SvmA=$PSA" >> "$WLOG"

# CPU jobs
[ "$PRF" -gt 0 ] && ! alive "$LOGD/.RF.pid"   && start_win RF   4 20
[ "$PSW" -gt 0 ] && ! alive "$LOGD/.SvmW.pid" && start_win SvmW 4 50

# GPU jobs (one at a time): Lstm (fast) first, then SvmA
if ! gpu_busy; then
  if   [ "$PLS" -gt 0 ]; then start_wsl Lstm "$LSTMPY" 3 ""
  elif [ "$PSA" -gt 0 ]; then start_wsl SvmA "$SVMAPY" 1 "SVMA_USE_CUML=1 SVMA_PSO_PROCESSES=1"
  fi
fi

# all complete?
if [ "$PRF" -eq 0 ] && [ "$PSW" -eq 0 ] && [ "$PLS" -eq 0 ] && [ "$PSA" -eq 0 ]; then
  echo "$(ts) [c1] ALL COMPLETE — removing scheduled task" >> "$WLOG"
  schtasks //delete //tn C1_Watchdog //f >/dev/null 2>&1
fi
