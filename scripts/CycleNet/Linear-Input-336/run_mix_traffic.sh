#!/bin/bash
# Ablation: base CycleNet vs CycleNetMix (gated channel-mixing branch)
# Prediction from the cross-variable dependency analysis:
#   - Traffic (Very strong dependency): CycleNetMix should IMPROVE over CycleNet
#   - ETTm1  (Very weak dependency):   CycleNetMix should be ~EQUAL to CycleNet
#     (gate stays near zero; check the learned |gate| in the checkpoint)
# Best lookbacks from the lookback study: Traffic=720, ETTm1=336.

seq_len_traffic=720
model_type=mlp   # also repeat with 'linear' if time allows

for pred_len in 96 192; do
  for model in CycleNetMix; do

    # ---------------- Traffic (hourly, weekly cycle=168) ----------------
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path traffic.csv \
      --model_id traffic_${seq_len_traffic}_${pred_len} \
      --model $model \
      --data custom \
      --features M \
      --seq_len $seq_len_traffic \
      --pred_len $pred_len \
      --enc_in 862 \
      --cycle 168 \
      --model_type $model_type \
      --mix_hidden 128 \
      --mix_layers 1 \
      --mix_dropout 0.1 \
      --train_epochs 30 \
      --patience 5 \
      --itr 1

  done
done