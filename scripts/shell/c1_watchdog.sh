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

# pool_healthy MODEL: return 0 (=treat pool as alive, SKIP reap/relaunch) when non-pooled
# train.py workers of MODEL already exist AND a cell log advanced recently. Protects
# orphaned-but-working pools -- the launcher dies on console-close (schtasks/nohup) while its
# train.py children keep running; the old unconditional reap+relaunch then restarted multi-day
# Mixed cells from Optuna trial 0 every churn cycle. Fail-safe: on any doubt, return 0 (don't reap).
pool_healthy(){
  local nw
  nw=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'train.py --model $1 ' -and \$_.CommandLine -notmatch 'pooled' }).Count" 2>/dev/null | tr -d '\r ')
  [ -z "$nw" ] && return 0                       # indeterminate (powershell hiccup) -> don't reap
  [ "$nw" -eq 0 ] 2>/dev/null && return 1         # no workers at all -> allow (re)launch
  # workers exist: healthy if any cell log advanced within 6h (tolerates slow Optuna trials;
  # worst observed inter-trial gap ~3.3h on heavy Mixed cells). Else assume hung -> allow relaunch.
  find "$LOGD" -maxdepth 1 -name "${1}_imbalv3_*.log" -mmin -360 2>/dev/null | grep -q . && return 0
  return 1
}

start_win(){ # $1=model $2=workers $3=ntrials
  cd "$REPO" || return
  # Reap orphaned c1 workers of this model before relaunching. The launcher periodically
  # dies with STATUS_CONTROL_C_EXIT (console-close of the schtasks/nohup shell) while its
  # subprocess.run train.py children keep running as orphans; without this, each relaunch
  # stacks +N more workers on the SAME pending cells -> CPU oversubscription (observed 3x /
  # 24 workers on 20 cores) that cripples the slow mixed SvmW cells. Excludes pooled (IV25).
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'train.py --model $1 ' -and \$_.CommandLine -notmatch 'pooled' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }" >/dev/null 2>&1
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 N_TRIALS_OVERRIDE="$3" PYTHONPATH=. \
    nohup "$WINPY" "$LAUNCH" --model "$1" --workers "$2" >> "$LOGD/_run_$1.log" 2>&1 &
  echo $! > "$LOGD/.$1.pid"; echo "$(ts) [c1] launched $1 (win, pid $!, orphans reaped)" >> "$WLOG"
}
start_wsl(){ # $1=model $2=venvpy $3=workers $4=extraenv $5=limit(optional)
  local LIM=""; [ -n "$5" ] && LIM="--limit $5"
  nohup wsl -e bash -lc "cd $WREPO && CUDA_VISIBLE_DEVICES=0 $4 PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=3 $2 $LAUNCH --model $1 --workers $3 $LIM" >> "$LOGD/_run_$1.log" 2>&1 &
  echo $! > "$LOGD/.$1.pid"; echo "$(ts) [c1] launched $1 (wsl, pid $!)" >> "$WLOG"
}

# --- option-a top-up (2026-07-17): restore full worker count WITHOUT reaping healthy orphans ---
nworkers(){ powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'train.py --model $1 ' -and \$_.CommandLine -notmatch 'pooled' }).Count" 2>/dev/null | tr -d '\r '; }
# idle_pending MODEL: count of pending cells whose per-cell log is absent or stale (>10h) = NOT
# currently being trained (same 10h window as the launcher's cell_in_progress skip).
idle_pending(){
  local tag lp n=0
  while IFS= read -r tag; do
    [ -z "$tag" ] && continue
    lp="$LOGD/${1}_${tag}.log"
    if [ ! -f "$lp" ] || ! find "$lp" -mmin -600 2>/dev/null | grep -q .; then n=$((n+1)); fi
  done < <(PYTHONPATH=. "$WINPY" "$REPO/$LAUNCH" --model "$1" --dry-run 2>/dev/null | grep imbalv3)
  echo "$n"
}
# start_win_topup: launch ADDITIONAL workers WITHOUT reaping. The launcher's cell_in_progress skip
# (fresh per-cell log) keeps these from racing cells the surviving orphans are already training.
start_win_topup(){ # $1=model $2=workers $3=ntrials
  cd "$REPO" || return
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 N_TRIALS_OVERRIDE="$3" PYTHONPATH=. \
    nohup "$WINPY" "$LAUNCH" --model "$1" --workers "$2" >> "$LOGD/_run_$1_topup.log" 2>&1 &
  echo "$(ts) [c1] TOP-UP $1 (+$2 workers, no reap, in-progress-skip)" >> "$WLOG"
}

PRF=$(pend RF); PSW=$(pend SvmW); PLS=$(pend Lstm); PSA=$(pend SvmA)
echo "$(ts) [c1] pending: RF=$PRF SvmW=$PSW Lstm=$PLS SvmA=$PSA" >> "$WLOG"

# CPU jobs
[ "$PRF" -gt 0 ] && ! alive "$LOGD/.RF.pid"   && start_win RF   4 20
if [ "$PSW" -gt 0 ] && ! alive "$LOGD/.SvmW.pid"; then
  NWSW=$(nworkers SvmW)
  if ! printf '%s' "$NWSW" | grep -qE '^[0-9]+$'; then
    echo "$(ts) [c1] SvmW worker count indeterminate -> no action (fail-safe, no reap)" >> "$WLOG"
  elif [ "$NWSW" -eq 0 ]; then
    start_win SvmW 8 50                                   # confirmed no workers -> full (re)launch
  elif [ "$NWSW" -ge 8 ]; then
    :                                                     # at target, nothing to do
  else
    IDLE=$(idle_pending SvmW)
    if printf '%s' "$IDLE" | grep -qE '^[0-9]+$' && [ "$IDLE" -ge 1 ]; then
      ADD=$((8-NWSW)); [ "$ADD" -gt "$IDLE" ] && ADD="$IDLE"
      start_win_topup SvmW "$ADD" 50                      # top-up shortfall on idle cells (no reap)
    else
      echo "$(ts) [c1] SvmW $NWSW workers, no idle pending -> adopt orphans, no top-up" >> "$WLOG"
    fi
  fi
fi

# GPU jobs (one at a time): Lstm (fast) first, then SvmA
if ! gpu_busy; then
  if   [ "$PLS" -gt 0 ]; then start_wsl Lstm "$LSTMPY" 3 ""
  elif [ "$PSA" -gt 0 ]; then start_wsl SvmA "$SVMAPY" 1 "SVMA_USE_CUML=1 SVMA_PSO_PROCESSES=1 SVMA_PSO_MAXITER=30"
  fi
fi

# all complete?
if [ "$PRF" -eq 0 ] && [ "$PSW" -eq 0 ] && [ "$PLS" -eq 0 ] && [ "$PSA" -eq 0 ]; then
  echo "$(ts) [c1] ALL COMPLETE — removing scheduled task" >> "$WLOG"
  schtasks //delete //tn C1_Watchdog //f >/dev/null 2>&1
fi
