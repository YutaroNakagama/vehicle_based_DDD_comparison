#!/bin/bash
# Generate remaining SvmW job list for auto-resub daemon
# Output format: MODEL|CONDITION|MODE|DISTANCE|DOMAIN|SEED|RATIO|WALLTIME|MEM|N_TRIALS

OUT="/tmp/remaining_svmw_zhao2009.txt"
> "$OUT"

SEEDS=(42 123)
RATIOS=(0.1 0.5)
DISTANCES=(mmd dtw wasserstein)
DOMAINS=(out_domain in_domain)
MODES=(source_only target_only)
CONDITIONS_RATIO=(smote_plain smote undersample)

# Baseline (no ratio)
for DISTANCE in "${DISTANCES[@]}"; do
    for DOMAIN in "${DOMAINS[@]}"; do
        for MODE in "${MODES[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                echo "SvmW|baseline|$MODE|$DISTANCE|$DOMAIN|$SEED||12:00:00|16gb|100" >> "$OUT"
            done
        done
    done
done

# Ratio-based conditions
for COND in "${CONDITIONS_RATIO[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        for DISTANCE in "${DISTANCES[@]}"; do
            for DOMAIN in "${DOMAINS[@]}"; do
                for MODE in "${MODES[@]}"; do
                    for SEED in "${SEEDS[@]}"; do
                        echo "SvmW|$COND|$MODE|$DISTANCE|$DOMAIN|$SEED|$RATIO|12:00:00|16gb|100" >> "$OUT"
                    done
                done
            done
        done
    done
done

echo "Generated $(wc -l < "$OUT") jobs → $OUT"
