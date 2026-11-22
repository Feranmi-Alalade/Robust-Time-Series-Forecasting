#!/bin/bash

set -euo pipefail

# ============================================================
# CycleNetMixLowRank Multi-Seed Grid Search
#
# Tune:
#   mix_rank
#   mix_layers
#   learning_rate
#
# Fixed:
#   mix_dropout = 0.1
#
# Weather ranks: 4, 8, 16
# ETTm1 ranks:   2, 4, 6
#
# 18 configurations per dataset / horizon
# × 4 dataset/horizon settings
# = 72 configurations per seed
#
# Seeds:
#   2025
#   2026
#
# 72 × 2 = 144 new runs
# ============================================================

model_type=mlp
batch_size=64
mix_dropout=0.1


# ============================================================
# RANDOM SEEDS
# ============================================================

declare -a seeds=(
  2025
  2026
)


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
# COMMON HYPERPARAMETERS
# ============================================================

declare -a mix_layers_list=(
  1
  2
)

declare -a learning_rates=(
  0.0001
  0.0005
  0.001
)


# ============================================================
# MULTI-SEED EXPERIMENT LOOP
# ============================================================

for seed in "${seeds[@]}"; do

  echo ""
  echo "######################################################"
  echo "STARTING LOWRANK GRID SEARCH"
  echo "RANDOM SEED: $seed"
  echo "######################################################"
  echo ""


  for pred_len in 96 192; do

    for entry in "${datasets[@]}"; do

      IFS='|' read -r \
        data_path \
        data \
        data_name \
        enc_in \
        cycle \
        seq_len <<< "$entry"


      # ======================================================
      # DATASET-SPECIFIC RANKS
      # ======================================================

      if [[ "$data_name" == "weather" ]]; then

        mix_ranks=(
          4
          8
          16
        )

      elif [[ "$data_name" == "ETTm1" ]]; then

        mix_ranks=(
          2
          4
          6
        )

      else

        echo "ERROR: Unknown dataset $data_name"
        exit 1

      fi


      # ======================================================
      # LOWRANK GRID SEARCH
      # ======================================================

      for mix_rank in "${mix_ranks[@]}"; do

        for mix_layers in "${mix_layers_list[@]}"; do

          for learning_rate in "${learning_rates[@]}"; do


            echo "======================================================"
            echo "Dataset:          $data_name"
            echo "Prediction:       $pred_len"
            echo "Random seed:      $seed"
            echo "mix_rank:         $mix_rank"
            echo "mix_layers:       $mix_layers"
            echo "mix_dropout:      $mix_dropout"
            echo "learning_rate:    $learning_rate"
            echo "======================================================"


            python -u run.py \
              --is_training 1 \
              --root_path ./dataset/ \
              --data_path "$data_path" \
              --model_id "${data_name}_CycleNetMixLowRank_${seq_len}_${pred_len}_rank${mix_rank}_layers${mix_layers}_lr${learning_rate}" \
              --model CycleNetMixLowRank \
              --data "$data" \
              --features M \
              --seq_len "$seq_len" \
              --pred_len "$pred_len" \
              --enc_in "$enc_in" \
              --cycle "$cycle" \
              --model_type "$model_type" \
              --batch_size "$batch_size" \
              --mix_rank "$mix_rank" \
              --mix_layers "$mix_layers" \
              --mix_dropout "$mix_dropout" \
              --learning_rate "$learning_rate" \
              --train_epochs 30 \
              --patience 5 \
              --random_seed "$seed" \
              --itr 1

          done

        done

      done

    done

  done


  echo ""
  echo "######################################################"
  echo "COMPLETED LOWRANK GRID SEARCH"
  echo "RANDOM SEED: $seed"
  echo "######################################################"
  echo ""

done


echo "======================================================"
echo "ALL LOWRANK MULTI-SEED GRID SEARCHES COMPLETED"
echo "Seeds: 2025, 2026"
echo "======================================================"