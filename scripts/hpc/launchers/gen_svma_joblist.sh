#!/bin/bash
# Generate remaining SvmA job list for auto-resub daemon
# Output format: MODEL|CONDITION|MODE|DISTANCE|DOMAIN|SEED|RATIO|WALLTIME|MEM|N_TRIALS

OUT="/tmp/remaining_svma_arefnezhad2019.txt"
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
                echo "SvmA|baseline|$MODE|$DISTANCE|$DOMAIN|$SEED||24:00:00|32gb|100" >> "$OUT"
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
                            smote|smote_plain) WT="48:00:00" ;;
                            *)                 WT="24:00:00" ;;
                        esac
                        echo "SvmA|$COND|$MODE|$DISTANCE|$DOMAIN|$SEED|$RATIO|${WT}|32gb|100" >> "$OUT"
                    done
                done
            done
        done
    done
done

# ---- mixed (multi-domain): train on all 87 subjects, evaluate on each domain ----
# Increased resources: 30h/48gb (source_only: 24h/32gb)
for DISTANCE in "${DISTANCES[@]}"; do
    for DOMAIN in "${DOMAINS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            echo "SvmA|baseline|mixed|$DISTANCE|$DOMAIN|$SEED||30:00:00|48gb|100" >> "$OUT"
        done
    done
done

for COND in "${CONDITIONS_RATIO[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        for DISTANCE in "${DISTANCES[@]}"; do
            for DOMAIN in "${DOMAINS[@]}"; do
                for SEED in "${SEEDS[@]}"; do
                    case "$COND" in
                        smote|smote_plain) WT="48:00:00" ;;
                        *)                 WT="30:00:00" ;;
                    esac
                    echo "SvmA|$COND|mixed|$DISTANCE|$DOMAIN|$SEED|$RATIO|${WT}|48gb|100" >> "$OUT"
                done
            done
        done
    done
done

echo "Generated $(wc -l < "$OUT") jobs → $OUT"
