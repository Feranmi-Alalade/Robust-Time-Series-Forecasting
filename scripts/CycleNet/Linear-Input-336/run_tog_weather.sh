#!/bin/bash
# Ablation: base CycleNet vs CycleNetMix vs CycleNetMixLag, on Weather.
# Weather's own cross-variable dependency characterization (from the
# Granger + Transfer Entropy analysis) rated it "Very strong" overall,
# so the prediction here mirrors the Traffic case:
#   - CycleNetMix / CycleNetMixLag should IMPROVE over base CycleNet
#     if the mixing branch is exploiting real cross-channel structure.
# NOTE: seq_len=336 below is carried over from the ETTm1 lookback study,
# not from a Weather-specific lookback sweep -- no such sweep has been
# run yet for Weather, so treat this as a reasonable starting guess
# rather than an established best lookback.

seq_len_weather=336
model_type=mlp   # also repeat with 'linear' if time allows

for pred_len in 96 192; do
  for model in CycleNet CycleNetMix CycleNetMixLag; do

    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path weather.csv \
      --model_id weather_${model}_${seq_len_weather}_${pred_len} \
      --model $model \
      --data custom \
      --features M \
      --seq_len $seq_len_weather \
      --pred_len $pred_len \
      --enc_in 21 \
      --cycle 144 \
      --model_type $model_type \
      --mix_hidden 128 \
      --mix_layers 1 \
      --mix_dropout 0.1 \
      --train_epochs 30 \
      --patience 5 \
      --itr 1

  done
done