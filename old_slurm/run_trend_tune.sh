#!/bin/bash

set -euo pipefail

# ============================================================
# CycleNetTrend Hyperparameter Tuning
#
# Tune:
#   trend_kernel
#   trend_layers
#   learning_rate
#
# 3 trend kernels
# × 2 trend layer settings
# × 3 learning rates
# = 18 configurations per dataset / horizon
# ============================================================

model_type=mlp
batch_size=64


# ============================================================
# DATASETS
#
# Format:
#
# data_path | data | data_name | enc_in | cycle | seq_len
# ============================================================

declare -a datasets=(
  "weather.csv|custom|weather|21|144|336"
  "ETTm1.csv|ETTm1|ETTm1|7|96|336"
)


# ============================================================
# HYPERPARAMETER COMBINATIONS
#
# Format:
#
# trend_kernel | trend_layers | learning_rate
# ============================================================

declare -a combinations=(

  "13|1|0.0001"
  "13|1|0.0005"
  "13|1|0.001"

  "13|2|0.0001"
  "13|2|0.0005"
  "13|2|0.001"

  "25|1|0.0001"
  "25|1|0.0005"
  "25|1|0.001"

  "25|2|0.0001"
  "25|2|0.0005"
  "25|2|0.001"

  "49|1|0.0001"
  "49|1|0.0005"
  "49|1|0.001"

  "49|2|0.0001"
  "49|2|0.0005"
  "49|2|0.001"
)


# ============================================================
# EXPERIMENT LOOP
# ============================================================

for pred_len in 96 192; do

  for entry in "${datasets[@]}"; do

    IFS='|' read -r \
      data_path \
      data \
      data_name \
      enc_in \
      cycle \
      seq_len <<< "$entry"


    # ========================================================
    # CYCLENET TREND TUNING
    # ========================================================

    for combination in "${combinations[@]}"; do

      IFS='|' read -r \
        trend_kernel \
        trend_layers \
        learning_rate <<< "$combination"


      echo "======================================================"
      echo "Dataset:          $data_name"
      echo "Prediction:       $pred_len"
      echo "trend_kernel:     $trend_kernel"
      echo "trend_layers:     $trend_layers"
      echo "learning_rate:    $learning_rate"
      echo "======================================================"


      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path "$data_path" \
        --model_id "${data_name}_CycleNetTrend_${seq_len}_${pred_len}_tk${trend_kernel}_tl${trend_layers}_lr${learning_rate}" \
        --model CycleNetTrend \
        --data "$data" \
        --features M \
        --seq_len "$seq_len" \
        --pred_len "$pred_len" \
        --enc_in "$enc_in" \
        --cycle "$cycle" \
        --model_type "$model_type" \
        --batch_size "$batch_size" \
        --trend_kernel "$trend_kernel" \
        --trend_layers "$trend_layers" \
        --learning_rate "$learning_rate" \
        --train_epochs 30 \
        --patience 5 \
        --itr 1

    done

  done

done