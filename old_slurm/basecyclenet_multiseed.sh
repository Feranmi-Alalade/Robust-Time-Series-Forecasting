#!/bin/bash
set -euo pipefail

# ============================================================
# Base CycleNet multi-seed evaluation
# Seeds: 2024, 2025, 2026
# Datasets: Weather, ETTm1
# Lookback: 336
# Horizons: 96, 192
# Backbone: MLP
# ============================================================

model_type="mlp"
batch_size=64
learning_rate=0.0001

declare -a seeds=(
  2024
  2025
  2026
)

# data_path | data | data_name | enc_in | cycle | seq_len
declare -a datasets=(
  "weather.csv|custom|weather|21|144|336"
  "ETTm1.csv|ETTm1|ETTm1|7|96|336"
)

declare -a pred_lens=(
  96
  192
)

for seed in "${seeds[@]}"; do
  echo ""
  echo "######################################################"
  echo "STARTING BASE CYCLENET RUNS"
  echo "RANDOM SEED: $seed"
  echo "######################################################"
  echo ""

  for pred_len in "${pred_lens[@]}"; do
    for entry in "${datasets[@]}"; do
      IFS='|' read -r data_path data data_name enc_in cycle seq_len <<< "$entry"

      echo "======================================================"
      echo "Dataset:          $data_name"
      echo "Prediction:       $pred_len"
      echo "Lookback:         $seq_len"
      echo "Random seed:      $seed"
      echo "Learning rate:    $learning_rate"
      echo "======================================================"

      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path "$data_path" \
        --model_id "${data_name}_CycleNet_${seq_len}_${pred_len}_lr${learning_rate}" \
        --model CycleNet \
        --data "$data" \
        --features M \
        --seq_len "$seq_len" \
        --pred_len "$pred_len" \
        --enc_in "$enc_in" \
        --cycle "$cycle" \
        --model_type "$model_type" \
        --use_revin 1 \
        --batch_size "$batch_size" \
        --learning_rate "$learning_rate" \
        --train_epochs 30 \
        --patience 5 \
        --lradj type3 \
        --random_seed "$seed" \
        --itr 1

      echo ""
    done
  done

  echo "######################################################"
  echo "COMPLETED BASE CYCLENET RUNS"
  echo "RANDOM SEED: $seed"
  echo "######################################################"
  echo ""
done

echo "======================================================"
echo "ALL BASE CYCLENET MULTI-SEED RUNS COMPLETED"
echo "Seeds: 2024, 2025, 2026"
echo "Lookback: 336"
echo "Horizons: 96, 192"
echo "======================================================"