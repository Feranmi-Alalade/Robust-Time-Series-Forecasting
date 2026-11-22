#!/bin/bash
#SBATCH --job-name=cyclenet_mix_low_rank
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --output=low_rank_optune_%j.log
#SBATCH --error=low_rank_optune_%j.err

set -euo pipefail

VENV_DIR="/home/alalade/CycleNet/venv"

cd /home/alalade/CycleNet

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
  echo "Activated venv: $VENV_DIR"
else
  echo "WARNING: virtualenv not found at $VENV_DIR"
fi

bash low_rank_tuning_optuna.sh