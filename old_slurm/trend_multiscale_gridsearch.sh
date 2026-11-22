#!/bin/bash
set -euo pipefail

# CycleNetTrendMultiScale Grid Search
# Tune:
#   trend_layers
#   learning_rate
# Fixed:
#   trend_kernels = 25 49 97
#
# 6 configurations per dataset/horizon
# x 4 dataset/horizon settings
# = 24 configurations per seed
#
# Seeds: 2024, 2025, 2026
# 24 x 3 = 72 total runs

model_type=mlp
batch_size=64

declare -a seeds=(2024 2025 2026)
declare -a datasets=(
  "weather.csv|custom|weather|21|144|336"
  "ETTm1.csv|ETTm1|ETTm1|7|96|336"
)
declare -a trend_layers_list=(1 2)
declare -a learning_rates=(0.0001 0.0005 0.001)
declare -a trend_kernels=(25 49 97)

for seed in "${seeds[@]}"; do
  echo ""
  echo "######################################################"
  echo "STARTING MULTISCALE TREND GRID SEARCH"
  echo "RANDOM SEED: $seed"
  echo "######################################################"
  echo ""

  for pred_len in 96 192; do
    for entry in "${datasets[@]}"; do
      IFS='|' read -r data_path data data_name enc_in cycle seq_len <<< "$entry"

      for trend_layers in "${trend_layers_list[@]}"; do
        for learning_rate in "${learning_rates[@]}"; do
          echo "======================================================"
          echo "Dataset:          $data_name"
          echo "Prediction:       $pred_len"
          echo "Random seed:      $seed"
          echo "trend_kernels:    ${trend_kernels[*]}"
          echo "trend_layers:     $trend_layers"
          echo "learning_rate:    $learning_rate"
          echo "======================================================"

          python -u run.py \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path "$data_path" \
            --model_id "${data_name}_CycleNetTrendMultiScale_${seq_len}_${pred_len}_tk25-49-97_tl${trend_layers}_lr${learning_rate}" \
            --model CycleNetTrendMultiScale \
            --data "$data" \
            --features M \
            --seq_len "$seq_len" \
            --pred_len "$pred_len" \
            --enc_in "$enc_in" \
            --cycle "$cycle" \
            --model_type "$model_type" \
            --batch_size "$batch_size" \
            --trend_kernels "${trend_kernels[@]}" \
            --trend_layers "$trend_layers" \
            --learning_rate "$learning_rate" \
            --train_epochs 30 \
            --patience 5 \
            --random_seed "$seed" \
            --itr 1
        done
      done
    done
  done

  echo ""
  echo "######################################################"
  echo "COMPLETED MULTISCALE TREND GRID SEARCH"
  echo "RANDOM SEED: $seed"
  echo "######################################################"
  echo ""
done

echo "======================================================"
echo "ALL MULTISCALE TREND GRID SEARCHES COMPLETED"
echo "Seeds: 2024, 2025, 2026"
echo "======================================================"