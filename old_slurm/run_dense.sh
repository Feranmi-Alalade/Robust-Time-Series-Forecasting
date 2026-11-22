#!/bin/bash
# Compares CycleNetTrendMix (trend + LOW RANK mixing) against
# CycleNetTrendMixDense (trend + FULL DENSE mixing, same as CycleNetMix's
# mixing branch) across all three datasets. This directly tests whether
# the low rank constraint on the mixing branch is still necessary once
# the trend branch is also present, or whether the trend branch's
# influence on training is enough that the full dense mixing capacity is
# fine on its own.
#
# Expected pattern based on the standalone mixing ablation: dense mixing
# alone overfit on Traffic (worse than base CycleNet) and on ETTm1
# (slightly worse), but was still a genuine improvement on Weather. If
# that pattern holds here too, CycleNetTrendMixDense should underperform
# CycleNetTrendMix specifically on Traffic and ETTm1, and the two should
# be closer, or dense could even win, on Weather.

model_type=mlp
batch_size=64

declare -a datasets=(
  "traffic.csv|custom|traffic|862|168|720"
  "weather.csv|custom|weather|21|144|336"
  "ETTm1.csv|ETTm1|ETTm1|7|96|336"
)

for pred_len in 96 192; do
  for entry in "${datasets[@]}"; do
    IFS='|' read -r data_path data data_name enc_in cycle seq_len <<< "$entry"

    for model in CycleNetTrendMixDense; do
      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path $data_path \
        --model_id ${data_name}_${model}_${seq_len}_${pred_len} \
        --model $model \
        --data $data \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in $enc_in \
        --cycle $cycle \
        --model_type $model_type \
        --batch_size $batch_size \
        --mix_hidden 128 \
        --mix_rank 16 \
        --mix_layers 1 \
        --mix_dropout 0.1 \
        --moving_avg 25 \
        --train_epochs 30 \
        --patience 5 \
        --itr 1
    done
  done
done