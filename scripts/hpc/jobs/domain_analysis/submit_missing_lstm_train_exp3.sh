#!/bin/bash
# Auto-generated: submit missing Lstm exp3 TRAIN+EVAL jobs
# Missing conditions needing training: 38

echo "[SUBMIT] Lstm TRAIN | baseline | dtw | out_domain | s42"
qsub -N Ls_ba_do_dt_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=baseline,DISTANCE=dtw,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | baseline | mmd | in_domain | s42"
qsub -N Ls_ba_mi_dt_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=baseline,DISTANCE=mmd,DOMAIN=in_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | baseline | mmd | out_domain | s42"
qsub -N Ls_ba_mo_dt_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=baseline,DISTANCE=mmd,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | baseline | mmd | out_domain | s123"
qsub -N Ls_ba_mo_dt_s123 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=baseline,DISTANCE=mmd,DOMAIN=out_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote | dtw | out_domain | s42_r0.1"
qsub -N Ls_sm_do_dt_r0.1_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote,DISTANCE=dtw,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote | mmd | in_domain | s42_r0.1"
qsub -N Ls_sm_mi_dt_r0.1_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote,DISTANCE=mmd,DOMAIN=in_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote | mmd | in_domain | s42_r0.5"
qsub -N Ls_sm_mi_dt_r0.5_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote,DISTANCE=mmd,DOMAIN=in_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote | mmd | in_domain | s123_r0.5"
qsub -N Ls_sm_mi_dt_r0.5_s123 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote,DISTANCE=mmd,DOMAIN=in_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote | mmd | out_domain | s42_r0.1"
qsub -N Ls_sm_mo_dt_r0.1_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote,DISTANCE=mmd,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote | mmd | out_domain | s42_r0.5"
qsub -N Ls_sm_mo_dt_r0.5_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote,DISTANCE=mmd,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote | mmd | out_domain | s123_r0.1"
qsub -N Ls_sm_mo_dt_r0.1_s123 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote,DISTANCE=mmd,DOMAIN=out_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote | mmd | out_domain | s123_r0.5"
qsub -N Ls_sm_mo_dt_r0.5_s123 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote,DISTANCE=mmd,DOMAIN=out_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote_plain | dtw | in_domain | s42_r0.1"
qsub -N Ls_sm_di_dt_r0.1_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote_plain,DISTANCE=dtw,DOMAIN=in_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote_plain | dtw | in_domain | s123_r0.1"
qsub -N Ls_sm_di_dt_r0.1_s123 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote_plain,DISTANCE=dtw,DOMAIN=in_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote_plain | dtw | out_domain | s42_r0.1"
qsub -N Ls_sm_do_dt_r0.1_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote_plain,DISTANCE=dtw,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote_plain | dtw | out_domain | s42_r0.5"
qsub -N Ls_sm_do_dt_r0.5_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote_plain,DISTANCE=dtw,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote_plain | dtw | out_domain | s123_r0.1"
qsub -N Ls_sm_do_dt_r0.1_s123 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote_plain,DISTANCE=dtw,DOMAIN=out_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote_plain | mmd | in_domain | s42_r0.1"
qsub -N Ls_sm_mi_dt_r0.1_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote_plain,DISTANCE=mmd,DOMAIN=in_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote_plain | mmd | in_domain | s42_r0.5"
qsub -N Ls_sm_mi_dt_r0.5_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote_plain,DISTANCE=mmd,DOMAIN=in_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote_plain | mmd | in_domain | s123_r0.1"
qsub -N Ls_sm_mi_dt_r0.1_s123 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote_plain,DISTANCE=mmd,DOMAIN=in_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote_plain | mmd | out_domain | s42_r0.1"
qsub -N Ls_sm_mo_dt_r0.1_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote_plain,DISTANCE=mmd,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote_plain | mmd | out_domain | s42_r0.5"
qsub -N Ls_sm_mo_dt_r0.5_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote_plain,DISTANCE=mmd,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote_plain | mmd | out_domain | s123_r0.1"
qsub -N Ls_sm_mo_dt_r0.1_s123 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote_plain,DISTANCE=mmd,DOMAIN=out_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote_plain | mmd | out_domain | s123_r0.5"
qsub -N Ls_sm_mo_dt_r0.5_s123 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote_plain,DISTANCE=mmd,DOMAIN=out_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote_plain | wasserstein | in_domain | s123_r0.1"
qsub -N Ls_sm_wi_dt_r0.1_s123 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote_plain,DISTANCE=wasserstein,DOMAIN=in_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | smote_plain | wasserstein | out_domain | s123_r0.1"
qsub -N Ls_sm_wo_dt_r0.1_s123 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=smote_plain,DISTANCE=wasserstein,DOMAIN=out_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | undersample | dtw | in_domain | s123_r0.1"
qsub -N Ls_un_di_dt_r0.1_s123 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=undersample,DISTANCE=dtw,DOMAIN=in_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | undersample | dtw | out_domain | s42_r0.1"
qsub -N Ls_un_do_dt_r0.1_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=undersample,DISTANCE=dtw,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | undersample | dtw | out_domain | s42_r0.5"
qsub -N Ls_un_do_dt_r0.5_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=undersample,DISTANCE=dtw,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | undersample | dtw | out_domain | s123_r0.1"
qsub -N Ls_un_do_dt_r0.1_s123 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=undersample,DISTANCE=dtw,DOMAIN=out_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | undersample | mmd | in_domain | s42_r0.1"
qsub -N Ls_un_mi_dt_r0.1_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=undersample,DISTANCE=mmd,DOMAIN=in_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | undersample | mmd | in_domain | s42_r0.5"
qsub -N Ls_un_mi_dt_r0.5_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=undersample,DISTANCE=mmd,DOMAIN=in_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | undersample | mmd | in_domain | s123_r0.1"
qsub -N Ls_un_mi_dt_r0.1_s123 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=undersample,DISTANCE=mmd,DOMAIN=in_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | undersample | mmd | out_domain | s42_r0.1"
qsub -N Ls_un_mo_dt_r0.1_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=undersample,DISTANCE=mmd,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | undersample | mmd | out_domain | s42_r0.5"
qsub -N Ls_un_mo_dt_r0.5_s42 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=undersample,DISTANCE=mmd,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | undersample | mmd | out_domain | s123_r0.1"
qsub -N Ls_un_mo_dt_r0.1_s123 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=undersample,DISTANCE=mmd,DOMAIN=out_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | undersample | mmd | out_domain | s123_r0.5"
qsub -N Ls_un_mo_dt_r0.5_s123 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=undersample,DISTANCE=mmd,DOMAIN=out_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3

echo "[SUBMIT] Lstm TRAIN | undersample | wasserstein | in_domain | s123_r0.1"
qsub -N Ls_un_wi_dt_r0.1_s123 -l select=1:ncpus=8:ngpus=1:mem=8gb -l walltime=20:00:00 -q GPU-1A -v MODEL=Lstm,CONDITION=undersample,DISTANCE=wasserstein,DOMAIN=in_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.1 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
sleep 0.3
