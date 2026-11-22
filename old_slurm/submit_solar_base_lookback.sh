#!/bin/bash
#SBATCH --job-name=solar_base_lb
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=solar_base_lb_%j.log
#SBATCH --error=solar_base_lb_%j.err

set -euo pipefail

cd /home/alalade/CycleNet

source venv/bin/activate

bash run_solar_base_lookback.sh