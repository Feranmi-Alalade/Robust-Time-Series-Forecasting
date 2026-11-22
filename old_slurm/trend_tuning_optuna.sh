#!/bin/bash

set -euo pipefail

n_trials=50
model_type=mlp
batch_size=64

declare -a datasets=(
  "weather.csv|custom|weather|21|144|336"
  "ETTm1.csv|ETTm1|ETTm1|7|96|336"
)

for pred_len in 96 192; do

  for entry in "${datasets[@]}"; do

    IFS='|' read -r data_path data data_name enc_in cycle seq_len <<< "$entry"

    echo ""
    echo "======================================================"
    echo "Starting CycleNetTrend Broad Optuna TPE study"
    echo "Dataset:          $data_name"
    echo "Prediction:       $pred_len"
    echo "Sequence length:  $seq_len"
    echo "Channels:         $enc_in"
    echo "Cycle:            $cycle"
    echo "Trials:           $n_trials"
    echo "======================================================"

    python -u tune_trend_optuna.py \
      --data_name "$data_name" \
      --data_path "$data_path" \
      --data "$data" \
      --enc_in "$enc_in" \
      --cycle "$cycle" \
      --seq_len "$seq_len" \
      --pred_len "$pred_len" \
      --model_type "$model_type" \
      --batch_size "$batch_size" \
      --n_trials "$n_trials"

  done
done

echo ""
echo "======================================================"
echo "ALL CYCLENETTREND BROAD OPTUNA STUDIES COMPLETED"
echo "======================================================"