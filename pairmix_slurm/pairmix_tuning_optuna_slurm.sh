#!/bin/bash
#SBATCH --job-name=pairmix_optuna
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --output=pairmix_optuna_%j.log
#SBATCH --error=pairmix_optuna_%j.err

set -euo pipefail

VENV_DIR="/home/alalade/CycleNet/venv"

cd /home/alalade/CycleNet

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
  echo "Activated venv: $VENV_DIR"
else
  echo "WARNING: virtualenv not found at $VENV_DIR"
fi

bash pairmix_tuning_optuna.sh