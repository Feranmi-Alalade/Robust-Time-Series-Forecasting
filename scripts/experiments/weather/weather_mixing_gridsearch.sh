#!/bin/bash
set -euo pipefail

# Weather mixing grid search
# Run through: sbatch scripts/experiments/weather/weather_mixing_slurm.sh

PROJECT_DIR="${PROJECT_DIR:-/home/alalade/CycleNet}"
cd "$PROJECT_DIR"

SEED=2024
CYCLE=144
ENC_IN=21

RESULTS_DIR="weather_grid_results_full"
DEPENDENCY_PATH="${DEPENDENCY_PATH:-./dependency_scores/weather_dependency.npy}"
DRY_RUN="${DRY_RUN:-0}"

SEQ_LENS=(96 192 336 720)
PRED_LENS=(96 192 336 720)
LEARNING_RATES=(0.0001 0.0005 0.001 0.002)

# ============================================================
# 1. CHECK DATASET AND DEPENDENCY MATRIX
# ============================================================

if [[ "$DRY_RUN" != 1 ]]; then

    [[ -f run.py ]] || {
        echo "ERROR: run.py not found"
        exit 1
    }

    [[ -f summarize_mixing_grid.py ]] || {
        echo "ERROR: summarize_mixing_grid.py not found"
        exit 1
    }

    [[ -f ./dataset/weather.csv ]] || {
        echo "ERROR: Weather dataset not found"
        exit 1
    }

    [[ -f "$DEPENDENCY_PATH" ]] || {
        echo "ERROR: Missing dependency matrix: $DEPENDENCY_PATH"
        exit 1
    }

    python - "$DEPENDENCY_PATH" <<'PY'
import sys
import numpy as np

path = sys.argv[1]
matrix = np.load(path, mmap_mode="r", allow_pickle=False)

if matrix.shape != (21, 21):
    raise SystemExit(
        f"Invalid dependency matrix shape: {matrix.shape}"
    )

if not np.isfinite(matrix).all():
    raise SystemExit("Dependency matrix has NaN or Inf")

print(
    f"Using existing PairMix dependency matrix: "
    f"{path}, shape={matrix.shape}",
    flush=True,
)
PY

    mkdir -p "$RESULTS_DIR"
fi

# ============================================================
# 2. COMMON TRAINING PARAMETERS
# ============================================================

COMMON=(
    --is_training 1
    --root_path ./dataset/
    --data_path weather.csv
    --data custom
    --features M
    --enc_in "$ENC_IN"
    --cycle "$CYCLE"
    --model_type mlp
    --d_model 512
    --use_revin 1
    --train_epochs 30
    --patience 5
    --batch_size 256
    --lradj type3
    --random_seed "$SEED"
    --itr 1
    --num_workers 4
)

# ============================================================
# 3. EXTRACT VALIDATION AND TEST METRICS
# ============================================================

report_trial() {
    local run_id="$1"
    local log="$2"

    python - "$run_id" "$log" <<'PY'
import sys
from pathlib import Path
from summarize_mixing_grid import parse_log

run_id = sys.argv[1]
log_path = Path(sys.argv[2])

try:
    epoch, val_mse, test_mse, test_mae = parse_log(log_path)
except (ValueError, OSError) as exc:
    raise SystemExit(
        f"ERROR: Missing or invalid results for "
        f"{run_id}: {exc}"
    )

print(
    f"VALIDATION RESULT | {run_id} | "
    f"best_epoch={epoch} | "
    f"best_val_mse={val_mse:.7f} | "
    f"test_mse={test_mse:.7f} | "
    f"test_mae={test_mae:.7f}",
    flush=True,
)
PY
}

# ============================================================
# 4. EXECUTE ONE EXPERIMENT
# ============================================================

TRIAL_COUNT=0

run_trial() {
    local run_id="$1"
    shift

    TRIAL_COUNT=$((TRIAL_COUNT + 1))

    if [[ "$DRY_RUN" == 1 ]]; then
        echo "DRY_RUN [$TRIAL_COUNT/704]: $run_id"
        return 0
    fi

    local log="$RESULTS_DIR/${run_id}.log"
    local done_marker="$RESULTS_DIR/${run_id}.done"

    if [[ -f "$done_marker" && -f "$log" ]]; then
        echo "SKIP COMPLETED [$TRIAL_COUNT/704]: $run_id"
        report_trial "$run_id" "$log"
        return 0
    fi

    rm -f "$done_marker"

    echo
    echo "======================================================"
    echo "RUN [$TRIAL_COUNT/704]: $run_id"
    echo "======================================================"

    python -u run.py \
        "${COMMON[@]}" \
        "$@" 2>&1 | tee "$log"

    report_trial "$run_id" "$log"

    touch "$done_marker"
}

echo "======================================================"
echo "WEATHER - FULL MIXING GRID SEARCH"
echo "======================================================"
echo "Lookbacks: 96 192 336 720"
echo "Prediction horizons: 96 192 336 720"
echo "Models: CycleNetMix, PairMix, LowRank"
echo "Total experiments: 704"
echo "Seed: $SEED"
echo "Selection: Lowest validation MSE"
echo "======================================================"

# ============================================================
# 5. CYCLENET MIX
# 4 L x 4 H x 2 layers x 4 LR = 128 experiments
# ============================================================

echo "======= WEATHER - CYCLENET MIX (128) ======="

for SEQ_LEN in "${SEQ_LENS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
        for MIX_LAYERS in 1 2; do
            for LR in "${LEARNING_RATES[@]}"; do

                RUN_ID="Weather_CycleNetMix_L${SEQ_LEN}_H${PRED_LEN}_layers${MIX_LAYERS}_lr${LR}"

                run_trial "$RUN_ID" \
                    --model_id "$RUN_ID" \
                    --model CycleNetMixTune \
                    --seq_len "$SEQ_LEN" \
                    --pred_len "$PRED_LEN" \
                    --mix_hidden 64 \
                    --mix_layers "$MIX_LAYERS" \
                    --mix_dropout 0.1 \
                    --learning_rate "$LR"

            done
        done
    done
done

# ============================================================
# 6. CYCLENET PAIRMIX
# 4 L x 4 H x 3 ratios x 4 LR = 192 experiments
# ============================================================

echo "======= WEATHER - PAIRMIX (192) ======="

for SEQ_LEN in "${SEQ_LENS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
        for MIX_RATIO in 0.2 0.4 0.6; do
            for LR in "${LEARNING_RATES[@]}"; do

                RUN_ID="Weather_CycleNetPairMix_L${SEQ_LEN}_H${PRED_LEN}_mr${MIX_RATIO}_lr${LR}"

                run_trial "$RUN_ID" \
                    --model_id "$RUN_ID" \
                    --model CycleNetPairMix \
                    --seq_len "$SEQ_LEN" \
                    --pred_len "$PRED_LEN" \
                    --dependency_score_path "$DEPENDENCY_PATH" \
                    --mix_ratio "$MIX_RATIO" \
                    --learning_rate "$LR"

            done
        done
    done
done

# ============================================================
# 7. CYCLENET LOWRANK
# 4 L x 4 H x 3 ranks x 2 layers x 4 LR = 384
# ============================================================

echo "======= WEATHER - LOWRANK (384) ======="

for SEQ_LEN in "${SEQ_LENS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
        for MIX_RANK in 16 32 64; do
            for MIX_LAYERS in 1 2; do
                for LR in "${LEARNING_RATES[@]}"; do

                    RUN_ID="Weather_CycleNetMixLowRank_L${SEQ_LEN}_H${PRED_LEN}_rank${MIX_RANK}_layers${MIX_LAYERS}_lr${LR}"

                    run_trial "$RUN_ID" \
                        --model_id "$RUN_ID" \
                        --model CycleNetMixLowRank \
                        --seq_len "$SEQ_LEN" \
                        --pred_len "$PRED_LEN" \
                        --mix_rank "$MIX_RANK" \
                        --mix_layers "$MIX_LAYERS" \
                        --mix_dropout 0.1 \
                        --learning_rate "$LR"

                done
            done
        done
    done
done

# ============================================================
# 8. AUTOMATIC SUMMARY
# ============================================================

if [[ "$TRIAL_COUNT" -ne 704 ]]; then
    echo "ERROR: Expected 704 experiments, got $TRIAL_COUNT"
    exit 1
fi

if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY RUN COMPLETE: 704 unique configurations enumerated"
    exit 0
fi

echo "======= ALL 704 WEATHER RUNS COMPLETED ======="

python -u summarize_mixing_grid.py Weather

echo "WEATHER GRID SEARCH AND SUMMARY COMPLETED"
