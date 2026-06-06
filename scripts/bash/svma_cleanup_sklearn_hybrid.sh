#!/usr/bin/env bash
# Remove the 8 sklearn-derived SvmA outputs (dtw_out_dom ratio=0.5)
# that were produced during the hybrid run, so the cuML 1W run produces
# a uniform library set across all 144 SvmA jobs.
set -u
EVAL=/mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison/results/outputs/evaluation/SvmA
MODEL=/mnt/c/git/work/vehicle_ddd_eval/vehicle_based_DDD_comparison/models/SvmA

TAGS=(
    dtw_out_domain_domain_train_split2_subjectwise_ratio0.5_s2024
    dtw_out_domain_domain_train_split2_subjectwise_ratio0.5_s1337
    dtw_out_domain_domain_train_split2_subjectwise_ratio0.5_s1000
    dtw_out_domain_domain_train_split2_subjectwise_ratio0.5_s777
    dtw_out_domain_domain_train_split2_subjectwise_ratio0.5_s512
    dtw_out_domain_domain_train_split2_subjectwise_ratio0.5_s256
    dtw_out_domain_domain_train_split2_subjectwise_ratio0.5_s99
    dtw_out_domain_domain_train_split2_subjectwise_ratio0.5_s7
)

for tag in "${TAGS[@]}"; do
    echo "--- ${tag} ---"
    while IFS= read -r path; do
        parent=$(dirname "$(dirname "$path")")
        if [ -d "$parent" ] && [[ "$parent" == "$EVAL"/* ]]; then
            echo "  rm eval: $(basename "$parent")"
            rm -rf "$parent"
        fi
    done < <(find "$EVAL" -name "*${tag}*" 2>/dev/null)

    while IFS= read -r path; do
        parent=$(dirname "$(dirname "$path")")
        if [ -d "$parent" ] && [[ "$parent" == "$MODEL"/* ]]; then
            echo "  rm model: $(basename "$parent")"
            rm -rf "$parent"
        fi
    done < <(find "$MODEL" -name "*${tag}*" 2>/dev/null)
done

echo ""
echo "=== Post-delete ==="
echo "within JSONs: $(find "$EVAL" -name '*_within.json' 2>/dev/null | wc -l) / 144"
echo "cross  JSONs: $(find "$EVAL" -name '*_cross.json'  2>/dev/null | wc -l) / 144"
echo "eval  jobid dirs: $(find "$EVAL" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
echo "model jobid dirs: $(find "$MODEL" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
