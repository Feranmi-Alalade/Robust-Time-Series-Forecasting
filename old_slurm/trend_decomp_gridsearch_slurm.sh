#!/bin/bash

#SBATCH --job-name=trenddecomp_tuning_gridsearch
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --output=trenddecomp_tuning_gridsearch_%j.log
#SBATCH --error=trenddecomp_tuning_gridsearch_%j.err

set -euo pipefail

VENV_DIR="/home/alalade/CycleNet/venv"

cd /home/alalade/CycleNet

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
  echo "Activated venv: $VENV_DIR"
else
  echo "WARNING: virtualenv not found at $VENV_DIR"
fi

bash trend_decomp_gridsearch.sh