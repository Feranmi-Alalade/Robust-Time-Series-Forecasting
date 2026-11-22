# #!/bin/bash

# set -euo pipefail

# # ============================================================
# # CycleNetPairMix Hyperparameter Tuning
# #
# # Tune:
# #   mix_ratio
# #   learning_rate
# #
# # 3 mixing ratios
# # × 3 learning rates
# # = 9 configurations per dataset / horizon
# # ============================================================

# model_type=mlp
# batch_size=64


# # ============================================================
# # DATASETS
# #
# # Format:
# #
# # data_path | data | data_name | enc_in | cycle | seq_len |
# # dependency_score_path
# # ============================================================

# declare -a datasets=(
#   "weather.csv|custom|weather|21|144|336|./dependency_scores/weather_dependency.npy"
#   "ETTm1.csv|ETTm1|ETTm1|7|96|336|./dependency_scores/ETTm1_dependency.npy"
# )


# # ============================================================
# # HYPERPARAMETER COMBINATIONS
# #
# # Format:
# #
# # mix_ratio | learning_rate
# # ============================================================

# declare -a combinations=(

#   "0.2|0.0001"
#   "0.2|0.0005"
#   "0.2|0.001"

#   "0.4|0.0001"
#   "0.4|0.0005"
#   "0.4|0.001"

#   "0.6|0.0001"
#   "0.6|0.0005"
#   "0.6|0.001"
# )


# # ============================================================
# # EXPERIMENT LOOP
# # ============================================================

# for pred_len in 96 192; do

#   for entry in "${datasets[@]}"; do

#     IFS='|' read -r \
#       data_path \
#       data \
#       data_name \
#       enc_in \
#       cycle \
#       seq_len \
#       dependency_score_path <<< "$entry"


#     # ========================================================
#     # CHECK DEPENDENCY MATRIX EXISTS
#     # ========================================================

#     if [[ ! -f "$dependency_score_path" ]]; then
#       echo "ERROR: dependency score matrix not found:"
#       echo "$dependency_score_path"
#       echo ""
#       echo "Run create_dependency_scores.py first."
#       exit 1
#     fi


#     # ========================================================
#     # CYCLETNET PAIRWISE MIXING TUNING
#     # ========================================================

#     for combination in "${combinations[@]}"; do

#       IFS='|' read -r \
#         mix_ratio \
#         learning_rate <<< "$combination"


#       echo "======================================================"
#       echo "Dataset:          $data_name"
#       echo "Prediction:       $pred_len"
#       echo "mix_ratio:        $mix_ratio"
#       echo "learning_rate:    $learning_rate"
#       echo "dependency file:  $dependency_score_path"
#       echo "======================================================"


#       python -u run.py \
#         --is_training 1 \
#         --root_path ./dataset/ \
#         --data_path "$data_path" \
#         --model_id "${data_name}_CycleNetPairMix_${seq_len}_${pred_len}_mr${mix_ratio}_lr${learning_rate}" \
#         --model CycleNetPairMix \
#         --data "$data" \
#         --features M \
#         --seq_len "$seq_len" \
#         --pred_len "$pred_len" \
#         --enc_in "$enc_in" \
#         --cycle "$cycle" \
#         --model_type "$model_type" \
#         --batch_size "$batch_size" \
#         --dependency_score_path "$dependency_score_path" \
#         --mix_ratio "$mix_ratio" \
#         --learning_rate "$learning_rate" \
#         --train_epochs 30 \
#         --patience 5 \
#         --itr 1

#     done

#   done

# done

#!/bin/bash

set -euo pipefail

# ============================================================
# CycleNetPairMix Hyperparameter Tuning
#
# Tune:
#   mix_ratio
#   learning_rate
#
# 3 mixing ratios
# × 3 learning rates
# = 9 configurations per dataset / horizon
#
# Repeat full grid for seeds:
#   2025
#   2026
# ============================================================

model_type=mlp
batch_size=64


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
# data_path | data | data_name | enc_in | cycle | seq_len |
# dependency_score_path
# ============================================================

declare -a datasets=(
  "weather.csv|custom|weather|21|144|336|./dependency_scores/weather_dependency.npy"
  "ETTm1.csv|ETTm1|ETTm1|7|96|336|./dependency_scores/ETTm1_dependency.npy"
)


# ============================================================
# HYPERPARAMETER COMBINATIONS
#
# Format:
#
# mix_ratio | learning_rate
# ============================================================

declare -a combinations=(

  "0.2|0.0001"
  "0.2|0.0005"
  "0.2|0.001"

  "0.4|0.0001"
  "0.4|0.0005"
  "0.4|0.001"

  "0.6|0.0001"
  "0.6|0.0005"
  "0.6|0.001"
)


# ============================================================
# EXPERIMENT LOOP
# ============================================================

for seed in "${seeds[@]}"; do

  echo "======================================================"
  echo "STARTING FULL PAIRMIX GRID SEARCH"
  echo "Random seed: $seed"
  echo "======================================================"

  for pred_len in 96 192; do

    for entry in "${datasets[@]}"; do

      IFS='|' read -r \
        data_path \
        data \
        data_name \
        enc_in \
        cycle \
        seq_len \
        dependency_score_path <<< "$entry"


      # ======================================================
      # CHECK DEPENDENCY MATRIX EXISTS
      # ======================================================

      if [[ ! -f "$dependency_score_path" ]]; then
        echo "ERROR: dependency score matrix not found:"
        echo "$dependency_score_path"
        echo ""
        echo "Run create_dependency_scores.py first."
        exit 1
      fi


      # ======================================================
      # CYCLENET PAIRWISE MIXING GRID SEARCH
      # ======================================================

      for combination in "${combinations[@]}"; do

        IFS='|' read -r \
          mix_ratio \
          learning_rate <<< "$combination"


        echo "======================================================"
        echo "Dataset:          $data_name"
        echo "Prediction:       $pred_len"
        echo "Random seed:      $seed"
        echo "mix_ratio:        $mix_ratio"
        echo "learning_rate:    $learning_rate"
        echo "dependency file:  $dependency_score_path"
        echo "======================================================"


        python -u run.py \
          --is_training 1 \
          --root_path ./dataset/ \
          --data_path "$data_path" \
          --model_id "${data_name}_CycleNetPairMix_${seq_len}_${pred_len}_mr${mix_ratio}_lr${learning_rate}_seed${seed}" \
          --model CycleNetPairMix \
          --data "$data" \
          --features M \
          --seq_len "$seq_len" \
          --pred_len "$pred_len" \
          --enc_in "$enc_in" \
          --cycle "$cycle" \
          --model_type "$model_type" \
          --batch_size "$batch_size" \
          --dependency_score_path "$dependency_score_path" \
          --mix_ratio "$mix_ratio" \
          --learning_rate "$learning_rate" \
          --train_epochs 30 \
          --patience 5 \
          --random_seed "$seed" \
          --itr 1

      done

    done

  done

done


echo "======================================================"
echo "ALL PAIRMIX GRID SEARCHES COMPLETED"
echo "Seeds: 2025, 2026"
echo "======================================================"