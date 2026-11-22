#!/usr/bin/env bash
#SBATCH --job-name=ett_dependency
#SBATCH --partition=STUD
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=ett_dependency_%A_%a.out
#SBATCH --error=ett_dependency_%A_%a.err

# Build CycleNet residual-based dependency matrices for PairMix.
# SLURM array mapping: 0=ETTm2, 1=ETTh1, 2=ETTh2.
# If no explicit pretrained CycleNet checkpoint is provided, this script
# trains a separate CycleNet baseline (L336 H96, RevIN=1) on each dataset.
# It never substitutes a checkpoint from a mixing variant.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/alalade/CycleNet}"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/venv/bin/activate"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" --version

MODE="${1:---run}"
if [[ "$MODE" != '--run' && "$MODE" != '--plan' ]]; then
  echo 'Usage: bash scripts/experiments/ett/ett_dependency_matrices.sh [--run | --plan]' >&2
  exit 2
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "$TASK_ID" in
  0)
    DATASET='ETTm2'; DATA_FILE='ETTm2.csv'; CYCLE=96
    GIVEN_CHECKPOINT="${ETTM2_CHECKPOINT:-}"
    ;;
  1)
    DATASET='ETTh1'; DATA_FILE='ETTh1.csv'; CYCLE=24
    GIVEN_CHECKPOINT="${ETTH1_CHECKPOINT:-}"
    ;;
  2)
    DATASET='ETTh2'; DATA_FILE='ETTh2.csv'; CYCLE=24
    GIVEN_CHECKPOINT="${ETTH2_CHECKPOINT:-}"
    ;;
  *) echo "Unknown SLURM_ARRAY_TASK_ID=$TASK_ID (expected 0,1,2)" >&2; exit 2 ;;
esac

ENC_IN=7
SEQ_LEN=336
PRED_LEN=96
REVIN=1
SEED=2024
BATCH_SIZE=64
NUM_WORKERS=4
LEARNING_RATE="${LEARNING_RATE:-0.001}"
MODEL_ID="${DATASET}_CycleNet_dependency_L${SEQ_LEN}_H${PRED_LEN}_RevIN${REVIN}"
CKPT_ROOT="${PROJECT_DIR}/dependency_checkpoints/${DATASET}"
TRAIN_MARKER="${CKPT_ROOT}/training_completed.ok"
OUTPUT="${PROJECT_DIR}/dependency_scores/${DATASET}_dependency.npy"
OVERWRITE="${OVERWRITE:-0}"    # set 1 to recreate an existing matrix
TRAIN_IF_MISSING="${TRAIN_IF_MISSING:-1}" # set 0 to require a supplied checkpoint

printf '\n===== %s PairMix dependency =====\n' "$DATASET"
printf 'data=%s | C=%s | cycle=%s | L=%s | H=%s | RevIN=%s\n' \
  "$DATA_FILE" "$ENC_IN" "$CYCLE" "$SEQ_LEN" "$PRED_LEN" "$REVIN"
echo "Output: $OUTPUT"

if [[ "$MODE" == '--plan' ]]; then
  echo "Pretrained checkpoint override: ${GIVEN_CHECKPOINT:-<none>}; train if missing: $TRAIN_IF_MISSING"
  exit 0
fi

[[ -f "./dataset/$DATA_FILE" ]] || { echo "Missing ./dataset/$DATA_FILE" >&2; exit 1; }
[[ -f ./create_dependency_scores.py ]] || { echo 'Missing create_dependency_scores.py' >&2; exit 1; }
mkdir -p "$CKPT_ROOT" "$(dirname "$OUTPUT")"

verify_matrix() {
  "$PYTHON_BIN" - "$OUTPUT" "$ENC_IN" <<'PY'
import sys
import numpy as np
path, c = sys.argv[1], int(sys.argv[2])
D = np.load(path, allow_pickle=False)
assert D.shape == (c,c), f'Wrong shape {D.shape}; expected {(c,c)}'
assert np.isfinite(D).all(), 'Matrix contains NaN or Inf'
assert np.allclose(D, D.T, atol=1e-6), 'Matrix is not symmetric'
assert np.allclose(np.diag(D), 0, atol=1e-6), 'Diagonal is not zero'
print('\nMatrix:', path)
print('Shape:', D.shape)
print('Min/max:', float(D.min()), float(D.max()))
print('Mean off diagonal:', float(D[np.triu_indices(c, k=1)].mean()))
print('Symmetric:', np.allclose(D, D.T, atol=1e-6))
print('Diagonal zero:', np.allclose(np.diag(D), 0, atol=1e-6))
print('Verification: PASSED')
PY
}

if [[ -f "$OUTPUT" && "$OVERWRITE" != 1 ]]; then
  echo "Existing output found; validating without overwriting: $OUTPUT"
  verify_matrix
  exit 0
fi

if [[ -n "$GIVEN_CHECKPOINT" ]]; then
  CHECKPOINT="$GIVEN_CHECKPOINT"
  [[ -f "$CHECKPOINT" ]] || { echo "Checkpoint does not exist: $CHECKPOINT" >&2; exit 1; }
  echo 'Using explicitly supplied checkpoint; make sure it was trained as CycleNet/MLP with matching settings.'
else
  # Use only a checkpoint from this exact, dedicated training run.
  CKPT_PATTERN="*${MODEL_ID}_CycleNet_${DATASET}_ftM_sl${SEQ_LEN}_pl${PRED_LEN}_cycle${CYCLE}_mlp_seed${SEED}*"
  CHECKPOINT=""
  if [[ -f "$TRAIN_MARKER" ]]; then
    mapfile -t CKPTS < <(find "$CKPT_ROOT" -type f -name checkpoint.pth -path "$CKPT_PATTERN" | sort)
    if [[ ${#CKPTS[@]} -eq 1 ]]; then
      CHECKPOINT="${CKPTS[0]}"
    elif [[ ${#CKPTS[@]} -gt 1 ]]; then
      echo 'Multiple candidate checkpoints for the same model ID; refusing to guess:' >&2
      printf '%s\n' "${CKPTS[@]}" >&2
      exit 1
    fi
  fi

  if [[ -z "$CHECKPOINT" ]]; then
    [[ "$TRAIN_IF_MISSING" == 1 ]] || {
      echo "No verified CycleNet checkpoint for $DATASET." >&2
      echo "Pass ${DATASET^^}_CHECKPOINT=/path/to/checkpoint.pth (ETTM2/ETTH1/ETTH2)" >&2
      exit 1
    }
    [[ -f ./run.py ]] || { echo 'Missing run.py, needed to train CycleNet.' >&2; exit 1; }
    rm -f "$TRAIN_MARKER"
    echo "Training an independent CycleNet/MLP checkpoint for $DATASET ..."
    "$PYTHON_BIN" -u run.py \
      --is_training 1 \
      --root_path ./dataset/ --data_path "$DATA_FILE" \
      --model_id "$MODEL_ID" --model CycleNet --data "$DATASET" \
      --features M --seq_len "$SEQ_LEN" --label_len 0 --pred_len "$PRED_LEN" \
      --enc_in "$ENC_IN" --cycle "$CYCLE" \
      --model_type mlp --d_model 512 --use_revin "$REVIN" \
      --train_epochs 30 --patience 5 --itr 1 \
      --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" --lradj type3 \
      --num_workers "$NUM_WORKERS" --random_seed "$SEED" \
      --checkpoints "$CKPT_ROOT/"

    mapfile -t CKPTS < <(find "$CKPT_ROOT" -type f -name checkpoint.pth -path "$CKPT_PATTERN" | sort)
    [[ ${#CKPTS[@]} -eq 1 ]] || {
      echo "Training finished but expected exactly one checkpoint, found ${#CKPTS[@]}." >&2
      printf '%s\n' "${CKPTS[@]:-}" >&2
      exit 1
    }
    CHECKPOINT="${CKPTS[0]}"
    touch "$TRAIN_MARKER"
  fi
fi

printf 'Using CycleNet checkpoint: %s\n' "$CHECKPOINT"
TMP_OUTPUT="${OUTPUT%.npy}.tmp_${SLURM_JOB_ID:-manual}_${TASK_ID}.npy"
trap 'rm -f "$TMP_OUTPUT"' EXIT

# Same checkpoint-driven residual scoring interface used for PEMS08.
# IMPORTANT: create_dependency_scores.py must collect residuals from TRAIN only,
# not from validation/test; verify this in your local implementation.
"$PYTHON_BIN" -u create_dependency_scores.py \
  --root_path ./dataset/ \
  --data_path "$DATA_FILE" \
  --data "$DATASET" \
  --features M \
  --seq_len "$SEQ_LEN" \
  --label_len 0 \
  --pred_len "$PRED_LEN" \
  --enc_in "$ENC_IN" \
  --cycle "$CYCLE" \
  --model_type mlp \
  --use_revin "$REVIN" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --checkpoint "$CHECKPOINT" \
  --output "$TMP_OUTPUT"

# Only replace the final matrix after output verification.
"$PYTHON_BIN" - "$TMP_OUTPUT" "$ENC_IN" <<'PY'
import numpy as np, sys
D = np.load(sys.argv[1], allow_pickle=False)
c = int(sys.argv[2])
assert D.shape == (c,c), f'Unexpected dependency shape: {D.shape}'
assert np.isfinite(D).all(), 'Dependency matrix contains NaN/Inf'
assert np.allclose(D, D.T, atol=1e-6), 'Dependency matrix must be symmetric'
assert np.allclose(np.diag(D), 0, atol=1e-6), 'Dependency diagonal must be zero'
PY
mv -f "$TMP_OUTPUT" "$OUTPUT"
trap - EXIT
verify_matrix
ls -lh "$OUTPUT"
echo "DONE: $DATASET"
