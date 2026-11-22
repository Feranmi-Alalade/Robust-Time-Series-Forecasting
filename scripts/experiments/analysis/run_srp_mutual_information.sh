#!/usr/bin/env bash
#SBATCH --job-name=srp_mi
#SBATCH --partition=STUD
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=srp_mi_%j.out
#SBATCH --error=srp_mi_%j.err
set -euo pipefail
cd /home/alalade/CycleNet
source venv/bin/activate
which python
python --version
# CPU-only statistical characterization; no GPU requested.
python -u srp_mutual_information_test.py \
  --data-dir ./dataset \
  --output-dir ./mi_srp_results \
  --bins 8 --sample-size 8000 --pair-cap 2500 --shift-controls 3 --seed 2024
