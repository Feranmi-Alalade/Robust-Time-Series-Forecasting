#!/bin/bash
#SBATCH --job-name=solar_dep_base
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=solar_dependency_base_%j.log
#SBATCH --error=solar_dependency_base_%j.err

set -euo pipefail

cd /home/alalade/CycleNet
source venv/bin/activate

bash solar_dependency_base.sh