#!/bin/bash
# Runs CycleNetTrendMix (both branches, independently gated) across all
# four dataset settings used so far, alongside base CycleNet as the
# reference point. After training, check both gate values in each
# checkpoint (trend_branch.gate and channel_mixing[0].gate) -- since both
# gates are independent and zero-initialized, this single run tells you
# whether trend-extrapolation, channel-mixing, both, or neither actually
# helped on each dataset, without needing to compare across four separate
# architectures.
#
# Expected pattern, per the standalone ablations so far:
#   Traffic  (weak trend 0.194, strong dependency): mix gate may open
#            similarly to CycleNetMixLowRank (still a net negative there);
#            trend gate should stay near zero.
#   Weather  (dependency "Very strong", trend not yet characterized here):
#            mix gate should open (matches CycleNetMix's standalone gain);
#            trend gate's behavior is untested territory.
#   ETTm1    (strong trend 0.734, weak dependency): trend gate should open;
#            mix gate should stay near zero (matches CycleNetMix's
#            standalone near-neutral result there).

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

    for model in CycleNet CycleNetTrendMix CycleNetMix CycleNetTrend CycleNetMixLowRank CycleNetMixLag; do
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