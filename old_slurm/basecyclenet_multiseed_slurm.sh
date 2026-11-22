#!/bin/bash

#SBATCH --job-name=base_cyclenet_multiseed
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --output=base_cyclenet_multiseed_%j.log
#SBATCH --error=base_cyclenet_multiseed_%j.err

set -euo pipefail

VENV_DIR="/home/alalade/CycleNet/venv"

cd /home/alalade/CycleNet

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  source "$VENV_DIR/bin/activate"
  echo "Activated venv: $VENV_DIR"
else
  echo "WARNING: virtualenv not found at $VENV_DIR"
fi

bash basecyclenet_multiseed.sh