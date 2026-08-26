#!/bin/bash

set -euo pipefail

# Weather
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --model_id weather_CycleNet_dependency \
  --model CycleNet \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 21 \
  --cycle 144 \
  --model_type mlp \
  --batch_size 64 \
  --train_epochs 30 \
  --patience 5 \
  --itr 1

# ETTm1
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_CycleNet_dependency \
  --model CycleNet \
  --data ETTm1 \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 7 \
  --cycle 96 \
  --model_type mlp \
  --batch_size 64 \
  --train_epochs 30 \
  --patience 5 \
  --itr 1