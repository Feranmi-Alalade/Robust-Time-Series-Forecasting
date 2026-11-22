#!/bin/bash
#SBATCH --job-name=SolarMixFull
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=solar_mixing.out
#SBATCH --open-mode=append

set -euo pipefail

cd /home/alalade/CycleNet
source venv/bin/activate

echo "START: Solar mixing grid | SLURM job ${SLURM_JOB_ID:-unknown}"
echo 'One sequential job; 704 trials; no SLURM array'

bash solar_mixing_gridsearch.sh

echo 'FINISHED: Solar grid search and validation-selected summary'
