#!/bin/bash
#SBATCH --job-name=cyclenet_base
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=cyclenet_baseline.out
#SBATCH --error=cyclenet_baseline.err
#SBATCH --open-mode=append

set -euo pipefail

VENV_DIR="/home/alalade/CycleNet/venv"

cd /home/alalade/CycleNet

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
  echo "Activated venv: $VENV_DIR"
else
  echo "ERROR: virtualenv not found at $VENV_DIR"
  exit 1
fi

echo "=============================================="
echo "BASE CYCLENET GRID SEARCH"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "=============================================="

bash cyclenet_base_gridsearch.sh

echo "=============================================="
echo "GRID SEARCH COMPLETED"
echo "=============================================="