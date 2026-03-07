#!/bin/bash
# Auto-generated: submit missing SvmA exp3 domain_train jobs
# Generated: $(date)
# Missing conditions: 18

echo "[SUBMIT] SvmA | smote | dtw | in_domain | s42_r0.5"
qsub -N Sv_sm_di_dt_r0.5_s42 -l select=1:ncpus=8:mem=32gb -l walltime=48:00:00 -q SINGLE -v MODEL=SvmA,CONDITION=smote,DISTANCE=dtw,DOMAIN=in_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh
sleep 0.3

echo "[SUBMIT] SvmA | smote | dtw | in_domain | s123_r0.5"
qsub -N Sv_sm_di_dt_r0.5_s123 -l select=1:ncpus=8:mem=32gb -l walltime=48:00:00 -q LONG -v MODEL=SvmA,CONDITION=smote,DISTANCE=dtw,DOMAIN=in_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh
sleep 0.3

echo "[SUBMIT] SvmA | smote | dtw | out_domain | s42_r0.5"
qsub -N Sv_sm_do_dt_r0.5_s42 -l select=1:ncpus=8:mem=32gb -l walltime=48:00:00 -q DEFAULT -v MODEL=SvmA,CONDITION=smote,DISTANCE=dtw,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh
sleep 0.3

echo "[SUBMIT] SvmA | smote | mmd | in_domain | s42_r0.5"
qsub -N Sv_sm_mi_dt_r0.5_s42 -l select=1:ncpus=8:mem=32gb -l walltime=48:00:00 -q SINGLE -v MODEL=SvmA,CONDITION=smote,DISTANCE=mmd,DOMAIN=in_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh
sleep 0.3

echo "[SUBMIT] SvmA | smote | mmd | out_domain | s123_r0.5"
qsub -N Sv_sm_mo_dt_r0.5_s123 -l select=1:ncpus=8:mem=32gb -l walltime=48:00:00 -q LONG -v MODEL=SvmA,CONDITION=smote,DISTANCE=mmd,DOMAIN=out_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh
sleep 0.3

echo "[SUBMIT] SvmA | smote | wasserstein | in_domain | s42_r0.5"
qsub -N Sv_sm_wi_dt_r0.5_s42 -l select=1:ncpus=8:mem=32gb -l walltime=48:00:00 -q DEFAULT -v MODEL=SvmA,CONDITION=smote,DISTANCE=wasserstein,DOMAIN=in_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh
sleep 0.3

echo "[SUBMIT] SvmA | smote | wasserstein | out_domain | s42_r0.5"
qsub -N Sv_sm_wo_dt_r0.5_s42 -l select=1:ncpus=8:mem=32gb -l walltime=48:00:00 -q SINGLE -v MODEL=SvmA,CONDITION=smote,DISTANCE=wasserstein,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh
sleep 0.3

echo "[SUBMIT] SvmA | smote | wasserstein | out_domain | s123_r0.5"
qsub -N Sv_sm_wo_dt_r0.5_s123 -l select=1:ncpus=8:mem=32gb -l walltime=48:00:00 -q LONG -v MODEL=SvmA,CONDITION=smote,DISTANCE=wasserstein,DOMAIN=out_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh
sleep 0.3

echo "[SUBMIT] SvmA | smote_plain | dtw | in_domain | s42_r0.5"
qsub -N Sv_sm_di_dt_r0.5_s42 -l select=1:ncpus=8:mem=32gb -l walltime=48:00:00 -q DEFAULT -v MODEL=SvmA,CONDITION=smote_plain,DISTANCE=dtw,DOMAIN=in_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh
sleep 0.3

echo "[SUBMIT] SvmA | smote_plain | dtw | in_domain | s123_r0.5"
qsub -N Sv_sm_di_dt_r0.5_s123 -l select=1:ncpus=8:mem=32gb -l walltime=48:00:00 -q SINGLE -v MODEL=SvmA,CONDITION=smote_plain,DISTANCE=dtw,DOMAIN=in_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh
sleep 0.3

echo "[SUBMIT] SvmA | smote_plain | dtw | out_domain | s42_r0.5"
qsub -N Sv_sm_do_dt_r0.5_s42 -l select=1:ncpus=8:mem=32gb -l walltime=48:00:00 -q LONG -v MODEL=SvmA,CONDITION=smote_plain,DISTANCE=dtw,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh
sleep 0.3

echo "[SUBMIT] SvmA | smote_plain | dtw | out_domain | s123_r0.5"
qsub -N Sv_sm_do_dt_r0.5_s123 -l select=1:ncpus=8:mem=32gb -l walltime=48:00:00 -q DEFAULT -v MODEL=SvmA,CONDITION=smote_plain,DISTANCE=dtw,DOMAIN=out_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh
sleep 0.3

echo "[SUBMIT] SvmA | smote_plain | mmd | in_domain | s42_r0.5"
qsub -N Sv_sm_mi_dt_r0.5_s42 -l select=1:ncpus=8:mem=32gb -l walltime=48:00:00 -q SINGLE -v MODEL=SvmA,CONDITION=smote_plain,DISTANCE=mmd,DOMAIN=in_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh
sleep 0.3

echo "[SUBMIT] SvmA | smote_plain | mmd | in_domain | s123_r0.5"
qsub -N Sv_sm_mi_dt_r0.5_s123 -l select=1:ncpus=8:mem=32gb -l walltime=48:00:00 -q LONG -v MODEL=SvmA,CONDITION=smote_plain,DISTANCE=mmd,DOMAIN=in_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh
sleep 0.3

echo "[SUBMIT] SvmA | smote_plain | mmd | out_domain | s42_r0.5"
qsub -N Sv_sm_mo_dt_r0.5_s42 -l select=1:ncpus=8:mem=32gb -l walltime=48:00:00 -q DEFAULT -v MODEL=SvmA,CONDITION=smote_plain,DISTANCE=mmd,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh
sleep 0.3

echo "[SUBMIT] SvmA | smote_plain | mmd | out_domain | s123_r0.5"
qsub -N Sv_sm_mo_dt_r0.5_s123 -l select=1:ncpus=8:mem=32gb -l walltime=48:00:00 -q SINGLE -v MODEL=SvmA,CONDITION=smote_plain,DISTANCE=mmd,DOMAIN=out_domain,SEED=123,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh
sleep 0.3

echo "[SUBMIT] SvmA | smote_plain | wasserstein | in_domain | s42_r0.5"
qsub -N Sv_sm_wi_dt_r0.5_s42 -l select=1:ncpus=8:mem=32gb -l walltime=48:00:00 -q LONG -v MODEL=SvmA,CONDITION=smote_plain,DISTANCE=wasserstein,DOMAIN=in_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh
sleep 0.3

echo "[SUBMIT] SvmA | smote_plain | wasserstein | out_domain | s42_r0.5"
qsub -N Sv_sm_wo_dt_r0.5_s42 -l select=1:ncpus=8:mem=32gb -l walltime=48:00:00 -q DEFAULT -v MODEL=SvmA,CONDITION=smote_plain,DISTANCE=wasserstein,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true,RATIO=0.5 /home/s2240011/git/ddd/vehicle_based_DDD_comparison/scripts/hpc/jobs/train/pbs_prior_research_unified.sh
sleep 0.3
