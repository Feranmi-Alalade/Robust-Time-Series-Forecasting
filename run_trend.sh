#!/bin/bash
# Runs ONLY CycleNetTrend (no base CycleNet, no mixing branch) across all
# three datasets tested so far. Base CycleNet results are not re-run here
# since they're already available from the CycleNetTrendMix ablation logs
# -- this script exists specifically to isolate the trend branch's own
# gate value without the mixing branch present, to test whether the small
# trend gate seen on ETTm1 in the combined model (-0.044) was caused by
# the mixing branch crowding it out, or whether it's small even alone.
#
# After these finish, run print_trend_gates.py (below) and compare each
# dataset's standalone trend gate against the corresponding value from
# the combined model:
#   Traffic: combined = 0.050062
#   Weather: combined = 0.063052
#   ETTm1:   combined = -0.043810
# If ETTm1's standalone gate is meaningfully larger in magnitude than
# 0.044, that points to crowding-out by the mixing branch. If it's
# similarly small, the backbone itself is likely already absorbing most
# of ETTm1's trend, independent of any branch.

model=CycleNetTrend
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
      --moving_avg 25 \
      --train_epochs 30 \
      --patience 5 \
      --itr 1
  done
done