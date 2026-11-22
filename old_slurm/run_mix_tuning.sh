#!/bin/bash

set -euo pipefail

# ============================================================
# CycleNetMix Hyperparameter Tuning
#
# Tune:
#   mix_hidden
#   mix_layers
#   learning_rate
#
# Keep fixed:
#   mix_dropout = 0.1
#
# 3 hidden sizes
# × 2 layer settings
# × 3 learning rates
# = 18 combinations
# ============================================================


model_type=mlp
batch_size=64

# Fixed setting
mix_dropout=0.1


# ============================================================
# DATASETS
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
# mix_hidden | mix_layers | learning_rate
#
# 3 × 2 × 3 = 18 combinations
# ============================================================

declare -a combinations=(

  # mix_hidden = 32
  "32|1|0.0001"
  "32|1|0.0005"
  "32|1|0.001"

  "32|2|0.0001"
  "32|2|0.0005"
  "32|2|0.001"


  # mix_hidden = 64
  "64|1|0.0001"
  "64|1|0.0005"
  "64|1|0.001"

  "64|2|0.0001"
  "64|2|0.0005"
  "64|2|0.001"


  # mix_hidden = 128
  "128|1|0.0001"
  "128|1|0.0005"
  "128|1|0.001"

  "128|2|0.0001"
  "128|2|0.0005"
  "128|2|0.001"
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
    # CYCLENET MIX TUNING
    # ========================================================

    for combination in "${combinations[@]}"; do

      IFS='|' read -r \
        mix_hidden \
        mix_layers \
        learning_rate <<< "$combination"


      echo "======================================================"
      echo "Dataset:       $data_name"
      echo "Prediction:    $pred_len"
      echo "mix_hidden:    $mix_hidden"
      echo "mix_layers:    $mix_layers"
      echo "mix_dropout:   $mix_dropout (fixed)"
      echo "learning_rate: $learning_rate"
      echo "======================================================"


      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path "$data_path" \
        --model_id "${data_name}_CycleNetMixTune_${seq_len}_${pred_len}_mh${mix_hidden}_ml${mix_layers}_lr${learning_rate}" \
        --model CycleNetMixTune \
        --data "$data" \
        --features M \
        --seq_len "$seq_len" \
        --pred_len "$pred_len" \
        --enc_in "$enc_in" \
        --cycle "$cycle" \
        --model_type "$model_type" \
        --batch_size "$batch_size" \
        --mix_hidden "$mix_hidden" \
        --mix_layers "$mix_layers" \
        --mix_dropout "$mix_dropout" \
        --learning_rate "$learning_rate" \
        --train_epochs 30 \
        --patience 5 \
        --itr 1

    done

  done

done