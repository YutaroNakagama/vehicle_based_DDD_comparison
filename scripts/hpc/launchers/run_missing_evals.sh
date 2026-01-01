#!/bin/bash
# 評価未実行のジョブを一括評価

PROJECT_ROOT="/home/s2240011/git/ddd/vehicle_based_DDD_comparison"
cd "$(dirname "$0")"

echo "============================================================"
echo "Running missing evaluations"
echo "Time: $(date)"
echo "============================================================"

MISSING_JOBS=(14618593 14618595 14618597 14618610 14618696 14618729 14618756 14618772 14618794 14618815 14618821 14618829 14618861 14618874 14618879 14618889 14618918 14618920 14618928 14618939 14618945 14618950)

for TRAIN_ID in "${MISSING_JOBS[@]}"; do
    subdir="$PROJECT_ROOT/models/RF/${TRAIN_ID}/${TRAIN_ID}[1]"
    model_file=$(ls "$subdir"/*_pooled_*.pkl 2>/dev/null | grep -v "scaler\|selected" | head -1)
    
    if [ -n "$model_file" ]; then
        # モデルファイル名からタグを抽出
        filename=$(basename "$model_file")
        # RF_pooled_imbal_v2_smote_ratio0_1_seed42_14618557_1.pkl -> imbal_v2_smote_ratio0_1_seed42
        TAG=$(echo "$filename" | sed 's/.*pooled_\(imbal_v2_[^_]*.*\)_'$TRAIN_ID'_1\.pkl/\1/')
        
        # モデルタイプを判定
        if echo "$filename" | grep -q "BalancedRF"; then
            MODEL="BalancedRF"
        elif echo "$filename" | grep -q "EasyEnsemble"; then
            MODEL="EasyEnsemble"
        else
            MODEL="RF"
        fi
        
        # シードを抽出
        SEED=$(echo "$TAG" | grep -oP 'seed\K\d+' || echo "42")
        
        echo "Submitting eval for $TRAIN_ID (TAG=$TAG, MODEL=$MODEL)"
        
        qsub -q DEFAULT \
            -l select=1:ncpus=2:mem=4gb \
            -l walltime=02:00:00 \
            -v PBS_O_WORKDIR="$PROJECT_ROOT",MODEL="$MODEL",TAG="$TAG",TRAIN_JOBID="$TRAIN_ID",SEED="$SEED" \
            pbs_evaluate.sh
        
        sleep 0.3
    fi
done

echo ""
echo "Done! Submitted ${#MISSING_JOBS[@]} evaluation jobs"
