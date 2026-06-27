#!/usr/bin/env bash
# Unified watchdog for the exp3 comparison runs — keeps all grids running to
# completion independent of any interactive session. Resume-safe.
#
# Manages TWO experiments x 4 models:
#   B1  (target_only + SW-SMOTE) : b1_compare_launcher.py     tag b1cmp_*    (6 cells/model)
#   IV25 (pooled, no imbalance)  : iv2025_baseline_launcher.py tag iv25base_* (3 cells/model)
#
# CPU jobs (RF, SvmW) launch freely when incomplete & not alive.
# GPU jobs (Lstm, SvmA) are serialized: at most ONE GPU job alive at a time,
# launched in priority order (B1 first, then IV25).
# Self-deletes the scheduled task when everything is complete.
set +e
REPO="/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison"
WREPO="/mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison"
cd "$REPO" || exit 1
B1LOG="$REPO/logs/exp3_b1cmp"; IVLOG="$REPO/logs/iv2025_base"; mkdir -p "$B1LOG" "$IVLOG"
WLOG="$B1LOG/_watchdog.log"
WINPY="/c/Users/ynakagama/AppData/Local/Programs/Python/Python311/python"
LSTMPY="/home/ynakagama/.venv_tf_gpu/bin/python"
SVMAPY="/home/ynakagama/.venv_svma_cuml/bin/python"
ts(){ date '+%Y-%m-%d %H:%M:%S'; }

# done counts -------------------------------------------------------------
b1_done(){ find "$REPO/results/outputs/evaluation/$1" -name "*b1cmp_$1*within.json" 2>/dev/null | wc -l | tr -d ' '; }
iv_done(){ find "$REPO/results/outputs/evaluation/$1" -name "eval_results_$1_pooled_iv25base_$1*.json" 2>/dev/null | wc -l | tr -d ' '; }
alive(){ local pf="$1"; [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null; }

# any GPU job (B1 or IV25, Lstm or SvmA) currently alive?
gpu_busy(){
  for pf in "$B1LOG/.Lstm.pid" "$B1LOG/.SvmA.pid" "$IVLOG/.Lstm.pid" "$IVLOG/.SvmA.pid"; do
    alive "$pf" && return 0
  done
  return 1
}

start_win(){ # $1=model $2=launcher $3=workers $4=ntrials $5=logdir $6=pidfile
  cd "$REPO" || return
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  N_TRIALS_OVERRIDE="$4" PYTHONPATH=. \
  nohup "$WINPY" "scripts/python/train/$2" --model "$1" --workers "$3" >> "$5/_run_$1.log" 2>&1 &
  echo $! > "$6"; echo "$(ts) [wd] launched $1 ($2, win, pid $!)" >> "$WLOG"
}
start_wsl(){ # $1=model $2=venvpy $3=launcher $4=workers $5=extraenv $6=logdir $7=pidfile
  nohup wsl -e bash -lc "cd $WREPO && CUDA_VISIBLE_DEVICES=0 $5 PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=3 $2 scripts/python/train/$3 --model $1 --workers $4" >> "$6/_run_$1.log" 2>&1 &
  echo $! > "$7"; echo "$(ts) [wd] launched $1 ($3, wsl, pid $!)" >> "$WLOG"
}

B1=(RF $(b1_done RF) SvmW $(b1_done SvmW) Lstm $(b1_done Lstm) SvmA $(b1_done SvmA))
IV=(RF $(iv_done RF) SvmW $(iv_done SvmW) Lstm $(iv_done Lstm) SvmA $(iv_done SvmA))
echo "$(ts) [wd] B1: RF=$(b1_done RF) SvmW=$(b1_done SvmW) Lstm=$(b1_done Lstm) SvmA=$(b1_done SvmA) | IV25: RF=$(iv_done RF) SvmW=$(iv_done SvmW) Lstm=$(iv_done Lstm) SvmA=$(iv_done SvmA)" >> "$WLOG"

# Expected totals (per convergence-optimised seed counts):
#   RF/SvmW/Lstm: B1 = 11 seeds x 2 ratios = 22; IV25 = 11.
#   SvmA (chance, capped at 6 seeds): B1 = 6 x 2 = 12; IV25 = 6.
B1N=22; IVN=6; SVMA_B1N=12; SVMA_IVN=6
# --- CPU jobs (free to run anytime) ---
[ "$(b1_done SvmW)" -lt $B1N ] && ! alive "$B1LOG/.SvmW.pid" && start_win SvmW b1_compare_launcher.py 6 50 "$B1LOG" "$B1LOG/.SvmW.pid"
[ "$(b1_done RF)"   -lt $B1N ] && ! alive "$B1LOG/.RF.pid"   && start_win RF   b1_compare_launcher.py 6 20 "$B1LOG" "$B1LOG/.RF.pid"
[ "$(iv_done RF)"   -lt $IVN ] && ! alive "$IVLOG/.RF.pid"   && start_win RF   iv2025_baseline_launcher.py 3 20 "$IVLOG" "$IVLOG/.RF.pid"
# Re-enabled (fix: SvmW/SvmA dirs populated from common -> pooled eval works).
[ "$(iv_done SvmW)" -lt $IVN ] && ! alive "$IVLOG/.SvmW.pid" && start_win SvmW iv2025_baseline_launcher.py 3 50 "$IVLOG" "$IVLOG/.SvmW.pid"

# --- GPU jobs (serialized, one at a time, priority B1 then IV25) ---
if ! gpu_busy; then
  if   [ "$(b1_done Lstm)" -lt $B1N ]; then start_wsl Lstm "$LSTMPY" b1_compare_launcher.py 4 "" "$B1LOG" "$B1LOG/.Lstm.pid"
  elif [ "$(b1_done SvmA)" -lt $SVMA_B1N ]; then start_wsl SvmA "$SVMAPY" b1_compare_launcher.py 1 "SVMA_USE_CUML=1 SVMA_PSO_PROCESSES=1" "$B1LOG" "$B1LOG/.SvmA.pid"
  elif [ "$(iv_done Lstm)" -lt $IVN ]; then start_wsl Lstm "$LSTMPY" iv2025_baseline_launcher.py 3 "" "$IVLOG" "$IVLOG/.Lstm.pid"
  elif [ "$(iv_done SvmA)" -lt $SVMA_IVN ]; then start_wsl SvmA "$SVMAPY" iv2025_baseline_launcher.py 1 "SVMA_USE_CUML=1 SVMA_PSO_PROCESSES=1" "$IVLOG" "$IVLOG/.SvmA.pid"
  fi
fi

# --- all complete? remove scheduled task ---
if [ "$(b1_done RF)" -ge $B1N ] && [ "$(b1_done SvmW)" -ge $B1N ] && [ "$(b1_done Lstm)" -ge $B1N ] && [ "$(b1_done SvmA)" -ge $SVMA_B1N ] \
   && [ "$(iv_done RF)" -ge $IVN ] && [ "$(iv_done SvmW)" -ge $IVN ] && [ "$(iv_done Lstm)" -ge $IVN ] && [ "$(iv_done SvmA)" -ge $SVMA_IVN ]; then
  echo "$(ts) [wd] ALL COMPLETE (B1 + IV25) — removing scheduled task" >> "$WLOG"
  schtasks //delete //tn B1_Watchdog //f >/dev/null 2>&1
fi
