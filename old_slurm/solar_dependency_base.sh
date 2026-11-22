#!/bin/bash

set -euo pipefail

cd /home/alalade/CycleNet

echo "Training Base CycleNet for Solar dependency extraction..."

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path solar_AL.txt \
  --model_id Solar_CycleNet_dependency \
  --model CycleNet \
  --data Solar \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 137 \
  --cycle 144 \
  --model_type mlp \
  --train_epochs 30 \
  --patience 5 \
  --use_revin 0 \
  --batch_size 64 \
  --learning_rate 0.002 \
  --random_seed 2024 \
  --itr 1

echo "Solar Base CycleNet dependency model completed."