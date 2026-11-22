#!/bin/bash
#SBATCH --job-name=pems08_dependency
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=pems08_dependency_%j.out
#SBATCH --error=pems08_dependency_%j.err

set -euo pipefail

cd /home/alalade/CycleNet
source venv/bin/activate

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path PEMS08.npz \
  --model_id PEMS08_CycleNet_dependency \
  --model CycleNet \
  --data PEMS \
  --features M \
  --seq_len 336 \
  --label_len 0 \
  --pred_len 96 \
  --enc_in 170 \
  --cycle 288 \
  --model_type mlp \
  --use_revin 1 \
  --train_epochs 30 \
  --patience 5 \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --lradj type3 \
  --num_workers 4 \
  --random_seed 2024 \
  --itr 1 \
  --checkpoints ./checkpoints/