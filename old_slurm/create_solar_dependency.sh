#!/bin/bash

set -euo pipefail

cd /home/alalade/CycleNet

mkdir -p dependency_scores

echo "Creating Solar dependency matrix..."

python create_dependency_scores.py \
  --root_path ./dataset/ \
  --data_path solar_AL.txt \
  --data Solar \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 137 \
  --cycle 144 \
  --model_type mlp \
  --checkpoint /home/alalade/CycleNet/checkpoints/Solar_CycleNet_dependency_CycleNet_Solar_ftM_sl336_pl96_cycle144_mlp_seed2024/checkpoint.pth \
  --output ./dependency_scores/Solar_dependency.npy

echo "Done."

ls -lh dependency_scores/Solar_dependency.npy