#!/bin/bash
#SBATCH --job-name=cyclenet_diff_tune2
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --output=mix_tune_%j.log
#SBATCH --error=mix_tune_%j.err

set -euo pipefail

VENV_DIR="/home/alalade/CycleNet/venv"

cd /home/alalade/CycleNet

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
  echo "Activated venv: $VENV_DIR"
else
  echo "WARNING: virtualenv not found at $VENV_DIR"
fi

bash run_mix_tuning.sh