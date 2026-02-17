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

# ---- source_only / target_only ----
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
                        case "$COND" in
                            smote|smote_plain) WT="24:00:00" ;;
                            *)                 WT="12:00:00" ;;
                        esac
                        echo "SvmW|$COND|$MODE|$DISTANCE|$DOMAIN|$SEED|$RATIO|${WT}|16gb|100" >> "$OUT"
                    done
                done
            done
        done
    done
done

# ---- mixed (multi-domain): 全87名で訓練、各ドメインで評価 ----
# リソース増量: 16h/24gb (source_only: 12h/16gb)
for DISTANCE in "${DISTANCES[@]}"; do
    for DOMAIN in "${DOMAINS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            echo "SvmW|baseline|mixed|$DISTANCE|$DOMAIN|$SEED||16:00:00|24gb|100" >> "$OUT"
        done
    done
done

for COND in "${CONDITIONS_RATIO[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        for DISTANCE in "${DISTANCES[@]}"; do
            for DOMAIN in "${DOMAINS[@]}"; do
                for SEED in "${SEEDS[@]}"; do
                    case "$COND" in
                        smote|smote_plain) WT="24:00:00" ;;
                        *)                 WT="16:00:00" ;;
                    esac
                    echo "SvmW|$COND|mixed|$DISTANCE|$DOMAIN|$SEED|$RATIO|${WT}|24gb|100" >> "$OUT"
                done
            done
        done
    done
done

echo "Generated $(wc -l < "$OUT") jobs → $OUT"
