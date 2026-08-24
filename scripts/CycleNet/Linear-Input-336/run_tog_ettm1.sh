#!/bin/bash
# Ablation: base CycleNet vs CycleNetMix (gated channel-mixing branch)
# Prediction from the cross-variable dependency analysis:
#   - Traffic (Very strong dependency): CycleNetMix should IMPROVE over CycleNet
#   - ETTm1  (Very weak dependency):   CycleNetMix should be ~EQUAL to CycleNet
#     (gate stays near zero; check the learned |gate| in the checkpoint)
# Best lookbacks from the lookback study: Traffic=720, ETTm1=336.

seq_len_ettm1=336
model_type=mlp   # also repeat with 'linear' if time allows

for pred_len in 96 192; do
  for model in CycleNet CycleNetMix CycleNetMixLag; do
    # ---------------- ETTm1 (15-min, daily cycle=96) ----------------
    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ETTm1.csv \
      --model_id ETTm1_${model}_${seq_len_ettm1}_${pred_len} \
      --model $model \
      --data ETTm1 \
      --features M \
      --seq_len $seq_len_ettm1 \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 96 \
      --model_type $model_type \
      --mix_hidden 32 \
      --mix_layers 1 \
      --mix_dropout 0.1 \
      --train_epochs 30 \
      --patience 5 \
      --itr 1

  done
done