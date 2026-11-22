#!/bin/bash
#SBATCH --job-name=pems08_dep
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=pems08_dep_%j.out
#SBATCH --error=pems08_dep_%j.err

set -euo pipefail

cd /home/alalade/CycleNet
source venv/bin/activate

# Verify Python environment
which python
python --version

mkdir -p dependency_scores

echo "Creating PEMS08 dependency matrix..."

# Locate the pretrained CycleNet checkpoint
CHECKPOINT=$(find ./checkpoints \
  -name checkpoint.pth \
  -path "*PEMS08_CycleNet_dependency*" \
  -print -quit)

if [[ -z "$CHECKPOINT" ]]; then
  echo "ERROR: PEMS08 checkpoint not found."
  exit 1
fi

echo "Using checkpoint: $CHECKPOINT"

# Generate residual-based dependency matrix
python -u create_dependency_scores.py \
  --root_path ./dataset/ \
  --data_path PEMS08.npz \
  --data PEMS \
  --features M \
  --seq_len 336 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 170 \
  --cycle 288 \
  --model_type mlp \
  --use_revin 1 \
  --batch_size 64 \
  --num_workers 4 \
  --checkpoint "$CHECKPOINT" \
  --output ./dependency_scores/PEMS08_dependency.npy

echo "Dependency matrix generated."

# Verify the output
python - <<'PY'
import numpy as np

path = "./dependency_scores/PEMS08_dependency.npy"
D = np.load(path, allow_pickle=False)

print("\n========== PEMS08 DEPENDENCY ==========")
print("Shape:", D.shape)
print("Minimum:", D.min())
print("Maximum:", D.max())
print("Mean (off-diagonal):", D[np.triu_indices_from(D, k=1)].mean())
print("Symmetric:", np.allclose(D, D.T))
print("Contains NaN/Inf:", not np.isfinite(D).all())

assert D.shape == (170, 170), f"Unexpected shape: {D.shape}"
assert np.isfinite(D).all(), "Non-finite dependency values"
assert np.allclose(D, D.T), "Matrix is not symmetric"
assert np.allclose(np.diag(D), 0), "Diagonal must be zero"

print("Verification: PASSED")
PY

echo ""
echo "Done."
ls -lh dependency_scores/PEMS08_dependency.npy