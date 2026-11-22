#!/bin/bash
#SBATCH --job-name=electricity_dep
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=electricity_dep_%j.out
#SBATCH --error=electricity_dep_%j.err

set -euo pipefail

cd /home/alalade/CycleNet

PYTHON=/home/alalade/CycleNet/venv/bin/python

DATASET=Electricity
DATA_FILE=electricity.csv
ENC_IN=321
CYCLE=168
SEQ_LEN=336
PRED_LEN=96
REVIN=1
SEED=2024
BATCH_SIZE=64
LR=0.001

MODEL_ID=Electricity_CycleNet_dependency

CKPT_ROOT="./dependency_checkpoints/Electricity"
OUTPUT="./dependency_scores/Electricity_dependency.npy"

mkdir -p "$CKPT_ROOT"
mkdir -p ./dependency_scores

echo "=========================================="
echo "ELECTRICITY PAIRMIX DEPENDENCY MATRIX"
echo "=========================================="

"$PYTHON" --version

# Verify required files
test -f "./dataset/$DATA_FILE"
test -f ./create_dependency_scores.py

# Verify number of channels
"$PYTHON" - <<'PY'
import pandas as pd

df = pd.read_csv("./dataset/electricity.csv")
channels = df.drop(columns=["date"]).shape[1]

assert channels == 321, f"Expected 321 channels, found {channels}"

print("Dataset shape:", df.shape)
print("Channels:", channels)
PY

# ==========================================
# STEP 1: TRAIN BASE CYCLENET
# ==========================================

echo "Training Base CycleNet..."

"$PYTHON" -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path "$DATA_FILE" \
    --model_id "$MODEL_ID" \
    --model CycleNet \
    --data custom \
    --features M \
    --seq_len "$SEQ_LEN" \
    --label_len 0 \
    --pred_len "$PRED_LEN" \
    --enc_in "$ENC_IN" \
    --cycle "$CYCLE" \
    --model_type mlp \
    --d_model 512 \
    --use_revin "$REVIN" \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --lradj type3 \
    --num_workers 4 \
    --random_seed "$SEED" \
    --checkpoints "$CKPT_ROOT/"

# Locate the trained checkpoint

mapfile -t CHECKPOINTS < <(
    find "$CKPT_ROOT" -type f \
        -name checkpoint.pth \
        -path "*${MODEL_ID}*"
)

if [[ ${#CHECKPOINTS[@]} -ne 1 ]]; then
    echo "Expected exactly one checkpoint."
    printf '%s\n' "${CHECKPOINTS[@]}"
    exit 1
fi

CHECKPOINT="${CHECKPOINTS[0]}"

echo "Using checkpoint: $CHECKPOINT"

# ==========================================
# STEP 2: GENERATE DEPENDENCY MATRIX
# ==========================================

echo "Calculating residual dependencies..."

"$PYTHON" -u create_dependency_scores.py \
    --root_path ./dataset/ \
    --data_path "$DATA_FILE" \
    --data custom \
    --features M \
    --seq_len "$SEQ_LEN" \
    --label_len 0 \
    --pred_len "$PRED_LEN" \
    --enc_in "$ENC_IN" \
    --cycle "$CYCLE" \
    --model_type mlp \
    --use_revin "$REVIN" \
    --batch_size "$BATCH_SIZE" \
    --num_workers 4 \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT"

# ==========================================
# STEP 3: VERIFY DEPENDENCY MATRIX
# ==========================================

"$PYTHON" - <<'PY'
import numpy as np

path = "./dependency_scores/Electricity_dependency.npy"

D = np.load(path, allow_pickle=False)

assert D.shape == (321, 321), "Incorrect matrix shape"
assert np.isfinite(D).all(), "NaN or Inf detected"
assert np.allclose(D, D.T, atol=1e-6), "Not symmetric"
assert np.allclose(np.diag(D), 0, atol=1e-6), "Nonzero diagonal"

off_diag = D[np.triu_indices(321, k=1)]

assert np.all(off_diag >= -1e-6)
assert np.all(off_diag <= 1 + 1e-6)

print("\n==========================================")
print("DEPENDENCY MATRIX VERIFIED")
print("==========================================")
print("File:", path)
print("Shape:", D.shape)
print("Minimum:", D.min())
print("Maximum:", D.max())
print("Mean dependency:", off_diag.mean())
print("Median dependency:", np.median(off_diag))
print("Std dependency:", off_diag.std())
print("Symmetric:", np.allclose(D, D.T))
print("Zero diagonal:", np.allclose(np.diag(D), 0))
print("Verification: PASSED")
PY

echo "Electricity dependency matrix completed."