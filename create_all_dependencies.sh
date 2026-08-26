#!/bin/bash

set -euo pipefail

cd /home/alalade/CycleNet

mkdir -p dependency_scores

echo "Creating Weather dependency matrix..."

python create_dependency_scores.py \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 21 \
  --cycle 144 \
  --model_type mlp \
  --checkpoint /home/alalade/CycleNet/checkpoints/weather_CycleNet_dependency_CycleNet_custom_ftM_sl336_pl96_cycle144_mlp_seed2024/checkpoint.pth \
  --output ./dependency_scores/weather_dependency.npy


echo "Creating ETTm1 dependency matrix..."

python create_dependency_scores.py \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --data ETTm1 \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 7 \
  --cycle 96 \
  --model_type mlp \
  --checkpoint /home/alalade/CycleNet/checkpoints/ETTm1_CycleNet_dependency_CycleNet_ETTm1_ftM_sl336_pl96_cycle96_mlp_seed2024/checkpoint.pth \
  --output ./dependency_scores/ETTm1_dependency.npy


echo "Done."

ls -lh dependency_scores/