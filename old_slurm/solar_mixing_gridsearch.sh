#!/bin/bash
set -euo pipefail

# The experiment script (NOT the SLURM submission script).
# Normally run by: sbatch solar_mixing_slurm.sh
PROJECT_DIR="${PROJECT_DIR:-/home/alalade/CycleNet}"
cd "$PROJECT_DIR"

SEED=2024
CYCLE=144
ENC_IN=137
RESULTS_DIR="results/solar/solar_grid_results_full"
DEPENDENCY_PATH="${DEPENDENCY_PATH:-./dependency_scores/Solar_dependency_L96.npy}"
DRY_RUN="${DRY_RUN:-0}"

SEQ_LENS=(96 192 336 720)
PRED_LENS=(96 192 336 720)
LEARNING_RATES=(0.0001 0.0005 0.001 0.002)

if [[ "$DRY_RUN" != 1 ]]; then
    [[ -f run.py ]] || { echo 'ERROR: run.py not found'; exit 1; }
    [[ -f summarize_mixing_grid.py ]] || { echo 'ERROR: summarize_mixing_grid.py not found'; exit 1; }
    [[ -f ./dataset/solar_AL.txt ]] || { echo 'ERROR: Solar data not found'; exit 1; }
    [[ -f "$DEPENDENCY_PATH" ]] || { echo "ERROR: missing PairMix matrix: $DEPENDENCY_PATH"; exit 1; }

    # Validate a fixed, existing 137 x 137 matrix. Never regenerate it.
    python - "$DEPENDENCY_PATH" <<'PY'
import sys
import numpy as np
p = sys.argv[1]
a = np.load(p, mmap_mode='r', allow_pickle=False)
if a.shape != (137, 137) or not np.isfinite(a).all():
    raise SystemExit(f'Invalid PairMix matrix at {p}: shape={a.shape}')
print(f'Using existing PairMix dependency matrix: {p}, shape={a.shape}', flush=True)
PY
    mkdir -p "$RESULTS_DIR"
fi

COMMON=(
    --is_training 1
    --root_path ./dataset/
    --data_path solar_AL.txt
    --data Solar
    --features M
    --enc_in "$ENC_IN"
    --cycle "$CYCLE"
    --model_type mlp
    --d_model 512
    --use_revin 0
    --train_epochs 30
    --patience 5
    --batch_size 64
    --lradj type3
    --random_seed "$SEED"
    --itr 1
    --num_workers 4
)

# Always validate the log before writing a completion marker.
# Returns the checkpoint's best validation MSE and the final test metrics.
report_trial() {
    local run_id="$1" log="$2"
    python - "$run_id" "$log" <<'PY'
import sys
from pathlib import Path
from summarize_mixing_grid import parse_log

run_id, log_path = sys.argv[1:]
try:
    epoch, val_mse, test_mse, test_mae = parse_log(Path(log_path))
except (ValueError, OSError) as exc:
    raise SystemExit(f"ERROR: missing/invalid validation or test results for {run_id}: {exc}")
print(
    f"VALIDATION RESULT | {run_id} | best_epoch={epoch} | "
    f"best_val_mse={val_mse:.7f} | test_mse={test_mse:.7f} | test_mae={test_mae:.7f}",
    flush=True,
)
PY
}

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
        # A corrupted marker/log must stop the job rather than silently pass.
        report_trial "$run_id" "$log"
        return 0
    fi

    rm -f "$done_marker"
    echo
    echo "======================================================"
    echo "RUN [$TRIAL_COUNT/704]: $run_id"
    echo "======================================================"
    # pipefail ensures a failed model run is not hidden by tee.
    python -u run.py "${COMMON[@]}" "$@" 2>&1 | tee "$log"

    # Only mark as completed if both metrics can actually be read.
    report_trial "$run_id" "$log"
    touch "$done_marker"
}

echo '======================================================'
echo 'SOLAR: 704 mixing grid experiments'
echo 'Lookbacks: 96 192 336 720 | Horizons: 96 192 336 720'
echo "PairMix dependency matrix: $DEPENDENCY_PATH"
echo 'Model selection: minimum validation MSE for each model/horizon'
echo '======================================================'

############################################################
# 1. CycleNetMix: 4 L x 4 H x 2 layers x 4 LR = 128
############################################################
echo '======= SOLAR - CYCLENET MIX (128 experiments) ======='
for SEQ_LEN in "${SEQ_LENS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
        for MIX_LAYERS in 1 2; do
            for LR in "${LEARNING_RATES[@]}"; do
                RUN_ID="Solar_CycleNetMix_L${SEQ_LEN}_H${PRED_LEN}_layers${MIX_LAYERS}_lr${LR}"
                run_trial "$RUN_ID" \
                    --model_id "$RUN_ID" --model CycleNetMixTune \
                    --seq_len "$SEQ_LEN" --pred_len "$PRED_LEN" \
                    --mix_hidden 64 --mix_layers "$MIX_LAYERS" \
                    --mix_dropout 0.1 --learning_rate "$LR"
            done
        done
    done
done

############################################################
# 2. PairMix: 4 L x 4 H x 3 ratios x 4 LR = 192
############################################################
echo '======= SOLAR - PAIRMIX (192 experiments) ======='
for SEQ_LEN in "${SEQ_LENS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
        for MIX_RATIO in 0.2 0.4 0.6; do
            for LR in "${LEARNING_RATES[@]}"; do
                RUN_ID="Solar_CycleNetPairMix_L${SEQ_LEN}_H${PRED_LEN}_mr${MIX_RATIO}_lr${LR}"
                run_trial "$RUN_ID" \
                    --model_id "$RUN_ID" --model CycleNetPairMix \
                    --seq_len "$SEQ_LEN" --pred_len "$PRED_LEN" \
                    --dependency_score_path "$DEPENDENCY_PATH" \
                    --mix_ratio "$MIX_RATIO" --learning_rate "$LR"
            done
        done
    done
done

############################################################
# 3. LowRank: 4 L x 4 H x 3 ranks x 2 layers x 4 LR = 384
############################################################
echo '======= SOLAR - LOWRANK (384 experiments) ======='
for SEQ_LEN in "${SEQ_LENS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
        for MIX_RANK in 16 32 64; do
            for MIX_LAYERS in 1 2; do
                for LR in "${LEARNING_RATES[@]}"; do
                    RUN_ID="Solar_CycleNetMixLowRank_L${SEQ_LEN}_H${PRED_LEN}_rank${MIX_RANK}_layers${MIX_LAYERS}_lr${LR}"
                    run_trial "$RUN_ID" \
                        --model_id "$RUN_ID" --model CycleNetMixLowRank \
                        --seq_len "$SEQ_LEN" --pred_len "$PRED_LEN" \
                        --mix_rank "$MIX_RANK" --mix_layers "$MIX_LAYERS" \
                        --mix_dropout 0.1 --learning_rate "$LR"
                done
            done
        done
    done
done

[[ "$TRIAL_COUNT" -eq 704 ]] || { echo "ERROR: expected 704 configurations, got $TRIAL_COUNT"; exit 1; }

if [[ "$DRY_RUN" == 1 ]]; then
    echo "DRY RUN COMPLETE: $TRIAL_COUNT unique configurations enumerated"
    exit 0
fi

echo '======= ALL 704 RUNS COMPLETED; SUMMARIZING ======='
python -u summarize_mixing_grid.py Solar

echo 'SOLAR GRID SEARCH AND VALIDATION-SELECTED SUMMARY COMPLETED'
