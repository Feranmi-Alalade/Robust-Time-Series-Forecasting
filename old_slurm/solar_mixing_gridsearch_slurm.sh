#!/bin/bash
#SBATCH --job-name=Solar_Mixing_L96
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=solar_mixing_L96_%j.out
#SBATCH --error=solar_mixing_L96_%j.err

set -euo pipefail

cd /home/alalade/CycleNet

source venv/bin/activate

echo "Running Solar mixing grid search"
echo "Lookback: 96"
echo "Horizons: 96, 192"
echo "Seed: 2024"

bash solar_mixing_gridsearch.sh
