#!/bin/bash
# Auto-generated: submit remaining 14 missing SvmA exp3 domain_train jobs
# Generated: 2025-03-02
# Missing: 4 imbalv3 (CONDITION=smote) + 10 smote_plain (CONDITION=smote_plain)
# All are ratio=0.5 conditions
# Queues: round-robin across SINGLE, DEFAULT, LONG

PBS_SCRIPT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh"
COMMON="-l select=1:ncpus=8:mem=32gb -l walltime=48:00:00"

# --- imbalv3 (CONDITION=smote, subjectwise) ---

echo "[SUBMIT] SvmA | smote(imbalv3) | dtw | in_domain | s42_r0.5"
qsub -N Sv_iv_di_dt_r05_s42 $COMMON -q SINGLE \
  -v MODEL=SvmA,CONDITION=smote,DISTANCE=dtw,DOMAIN=in_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 \
  "$PBS_SCRIPT"
sleep 0.3

echo "[SUBMIT] SvmA | smote(imbalv3) | dtw | in_domain | s123_r0.5"
qsub -N Sv_iv_di_dt_r05_s123 $COMMON -q DEFAULT \
  -v MODEL=SvmA,CONDITION=smote,DISTANCE=dtw,DOMAIN=in_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 \
  "$PBS_SCRIPT"
sleep 0.3

echo "[SUBMIT] SvmA | smote(imbalv3) | dtw | out_domain | s42_r0.5"
qsub -N Sv_iv_do_dt_r05_s42 $COMMON -q LONG \
  -v MODEL=SvmA,CONDITION=smote,DISTANCE=dtw,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 \
  "$PBS_SCRIPT"
sleep 0.3

echo "[SUBMIT] SvmA | smote(imbalv3) | wasserstein | out_domain | s123_r0.5"
qsub -N Sv_iv_wo_wa_r05_s123 $COMMON -q SINGLE \
  -v MODEL=SvmA,CONDITION=smote,DISTANCE=wasserstein,DOMAIN=out_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 \
  "$PBS_SCRIPT"
sleep 0.3

# --- smote_plain (CONDITION=smote_plain) ---

echo "[SUBMIT] SvmA | smote_plain | dtw | in_domain | s42_r0.5"
qsub -N Sv_sp_di_dt_r05_s42 $COMMON -q DEFAULT \
  -v MODEL=SvmA,CONDITION=smote_plain,DISTANCE=dtw,DOMAIN=in_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 \
  "$PBS_SCRIPT"
sleep 0.3

echo "[SUBMIT] SvmA | smote_plain | dtw | in_domain | s123_r0.5"
qsub -N Sv_sp_di_dt_r05_s123 $COMMON -q LONG \
  -v MODEL=SvmA,CONDITION=smote_plain,DISTANCE=dtw,DOMAIN=in_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 \
  "$PBS_SCRIPT"
sleep 0.3

echo "[SUBMIT] SvmA | smote_plain | dtw | out_domain | s42_r0.5"
qsub -N Sv_sp_do_dt_r05_s42 $COMMON -q SINGLE \
  -v MODEL=SvmA,CONDITION=smote_plain,DISTANCE=dtw,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 \
  "$PBS_SCRIPT"
sleep 0.3

echo "[SUBMIT] SvmA | smote_plain | dtw | out_domain | s123_r0.5"
qsub -N Sv_sp_do_dt_r05_s123 $COMMON -q DEFAULT \
  -v MODEL=SvmA,CONDITION=smote_plain,DISTANCE=dtw,DOMAIN=out_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 \
  "$PBS_SCRIPT"
sleep 0.3

echo "[SUBMIT] SvmA | smote_plain | mmd | in_domain | s42_r0.5"
qsub -N Sv_sp_mi_mm_r05_s42 $COMMON -q LONG \
  -v MODEL=SvmA,CONDITION=smote_plain,DISTANCE=mmd,DOMAIN=in_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 \
  "$PBS_SCRIPT"
sleep 0.3

echo "[SUBMIT] SvmA | smote_plain | mmd | in_domain | s123_r0.5"
qsub -N Sv_sp_mi_mm_r05_s123 $COMMON -q SINGLE \
  -v MODEL=SvmA,CONDITION=smote_plain,DISTANCE=mmd,DOMAIN=in_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 \
  "$PBS_SCRIPT"
sleep 0.3

echo "[SUBMIT] SvmA | smote_plain | mmd | out_domain | s42_r0.5"
qsub -N Sv_sp_mo_mm_r05_s42 $COMMON -q DEFAULT \
  -v MODEL=SvmA,CONDITION=smote_plain,DISTANCE=mmd,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 \
  "$PBS_SCRIPT"
sleep 0.3

echo "[SUBMIT] SvmA | smote_plain | mmd | out_domain | s123_r0.5"
qsub -N Sv_sp_mo_mm_r05_s123 $COMMON -q LONG \
  -v MODEL=SvmA,CONDITION=smote_plain,DISTANCE=mmd,DOMAIN=out_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 \
  "$PBS_SCRIPT"
sleep 0.3

echo "[SUBMIT] SvmA | smote_plain | wasserstein | in_domain | s42_r0.5"
qsub -N Sv_sp_wi_wa_r05_s42 $COMMON -q SINGLE \
  -v MODEL=SvmA,CONDITION=smote_plain,DISTANCE=wasserstein,DOMAIN=in_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 \
  "$PBS_SCRIPT"
sleep 0.3

echo "[SUBMIT] SvmA | smote_plain | wasserstein | out_domain | s42_r0.5"
qsub -N Sv_sp_wo_wa_r05_s42 $COMMON -q DEFAULT \
  -v MODEL=SvmA,CONDITION=smote_plain,DISTANCE=wasserstein,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 \
  "$PBS_SCRIPT"
sleep 0.3

echo ""
echo "=== 投入完了: 14 ジョブ (4 imbalv3 + 10 smote_plain) ==="
echo "キュー分散: SINGLE=5, DEFAULT=5, LONG=4"
