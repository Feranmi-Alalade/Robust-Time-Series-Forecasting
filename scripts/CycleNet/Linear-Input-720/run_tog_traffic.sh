#!/bin/bash
# Compares base CycleNet, dense mixing (CycleNetMix), and low-rank mixing
# (CycleNetMixLowRank) on Traffic at both horizons. mix_rank=16 gives an
# 86.5% reduction in mixing-branch parameters vs the dense CycleNetMix
# (223k -> 30k at Traffic's 862-channel scale) -- verified directly before
# this script was written. Worth also trying mix_rank=8 and mix_rank=32
# once this baseline comparison is in, to see how sensitive the result is
# to the exact rank chosen.

seq_len=720
cycle=168
model_type=mlp
batch_size=64   # matching the most recent Traffic ablation run

for pred_len in 96 192; do
  for model in CycleNet CycleNetMix CycleNetMixLowRank; do
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path traffic.csv \
      --model_id traffic_${model}_${seq_len}_${pred_len} \
      --model $model \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --cycle $cycle \
      --model_type $model_type \
      --batch_size $batch_size \
      --mix_hidden 128 \
      --mix_rank 16 \
      --mix_layers 1 \
      --mix_dropout 0.1 \
      --train_epochs 30 \
      --patience 5 \
      --itr 1
  done
done