#!/bin/bash
set -euo pipefail

# CycleNetTrendDecomp Grid Search
# Tune:
#   trend_kernel
#   trend_layers
#   learning_rate
#
# 18 configurations per dataset/horizon
# x 4 dataset/horizon settings
# = 72 total runs
#
# Seed: 2024

model_type=mlp
batch_size=64

seed=2024

declare -a datasets=(
  "weather.csv|custom|weather|21|144|336"
  "ETTm1.csv|ETTm1|ETTm1|7|96|336"
)
declare -a trend_kernels=(25 49 97)
declare -a trend_layers_list=(1 2)
declare -a learning_rates=(0.0001 0.0005 0.001)

echo ""
echo "######################################################"
echo "STARTING TREND DECOMP GRID SEARCH"
echo "RANDOM SEED: $seed"
echo "######################################################"
echo ""

for pred_len in 96 192; do
  for entry in "${datasets[@]}"; do
    IFS='|' read -r data_path data data_name enc_in cycle seq_len <<< "$entry"

    for trend_kernel in "${trend_kernels[@]}"; do
      for trend_layers in "${trend_layers_list[@]}"; do
        for learning_rate in "${learning_rates[@]}"; do
          echo "======================================================"
          echo "Dataset:          $data_name"
          echo "Prediction:       $pred_len"
          echo "Random seed:      $seed"
          echo "trend_kernel:     $trend_kernel"
          echo "trend_layers:     $trend_layers"
          echo "learning_rate:    $learning_rate"
          echo "======================================================"

          python -u run.py \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path "$data_path" \
            --model_id "${data_name}_CycleNetTrendDecomp_${seq_len}_${pred_len}_tk${trend_kernel}_tl${trend_layers}_lr${learning_rate}" \
            --model CycleNetTrendDecomp \
            --data "$data" \
            --features M \
            --seq_len "$seq_len" \
            --pred_len "$pred_len" \
            --enc_in "$enc_in" \
            --cycle "$cycle" \
            --model_type "$model_type" \
            --use_revin 1 \
            --batch_size "$batch_size" \
            --trend_kernel "$trend_kernel" \
            --trend_layers "$trend_layers" \
            --learning_rate "$learning_rate" \
            --train_epochs 30 \
            --patience 5 \
            --lradj type3 \
            --random_seed "$seed" \
            --itr 1
        done
      done
    done
  done
done

echo ""
echo "######################################################"
echo "TREND DECOMP GRID SEARCH COMPLETED"
echo "RANDOM SEED: $seed"
echo "######################################################"
echo ""