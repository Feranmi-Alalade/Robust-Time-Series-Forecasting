#!/bin/bash
#SBATCH --job-name=cy_mlp_36_fresh
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=cyclenet_mlp_restart_%j.out
#SBATCH --error=cyclenet_mlp_restart_%j.err

set -euo pipefail

cd /home/alalade/CycleNet
source venv/bin/activate

# Fresh restart: always run all 36 experiments.
# Keep checkpoints separate from previous jobs.
RUN_ID="${SLURM_JOB_ID:-manual}"

echo "SLURM Job ID: $RUN_ID"
echo "CycleNet/MLP | Seed: 2024"
echo "Lookbacks: 96, 336, 720"
echo "Horizons: 96, 192, 336, 720"
echo "Total experiments: 36"

completed=0

for dataset in ettm1 solar weather; do

  case "$dataset" in
    ettm1)
      data_name=ETTm1
      data_path=ETTm1.csv
      model_id=ETTm1
      enc_in=7
      cycle=96
      batch_size=256
      use_revin=1
      ;;
    solar)
      data_name=Solar
      data_path=solar_AL.txt
      model_id=Solar
      enc_in=137
      cycle=144
      batch_size=64
      use_revin=0
      ;;
    weather)
      data_name=custom
      data_path=weather.csv
      model_id=weather
      enc_in=21
      cycle=144
      batch_size=256
      use_revin=1
      ;;
  esac

  for seq_len in 96 336 720; do

    case "${dataset}:${seq_len}" in
      ettm1:96)   learning_rate=0.005 ;;
      ettm1:336)  learning_rate=0.002 ;;
      ettm1:720)  learning_rate=0.0005 ;;
      solar:96)   learning_rate=0.01 ;;
      solar:336)  learning_rate=0.002 ;;
      solar:720)  learning_rate=0.0002 ;;
      weather:96)  learning_rate=0.005 ;;
      weather:336) learning_rate=0.002 ;;
      weather:720) learning_rate=0.0005 ;;
    esac

    checkpoints="./paper_reproduction_checkpoints/restart_${RUN_ID}/${dataset}/L${seq_len}/"
    mkdir -p "$checkpoints"

    for pred_len in 96 192 336 720; do

      echo "================================================="
      echo "RUN $((completed + 1))/36"
      echo "Dataset: $dataset"
      echo "Lookback: $seq_len"
      echo "Horizon: $pred_len"
      echo "Learning rate: $learning_rate"
      echo "Batch size: $batch_size"
      echo "RevIN: $use_revin"
      echo "Seed: 2024"
      echo "================================================="

      python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path "$data_path" \
        --model_id "${model_id}_${seq_len}_${pred_len}" \
        --model CycleNet \
        --data "$data_name" \
        --features M \
        --seq_len "$seq_len" \
        --pred_len "$pred_len" \
        --enc_in "$enc_in" \
        --cycle "$cycle" \
        --model_type mlp \
        --use_revin "$use_revin" \
        --train_epochs 30 \
        --patience 5 \
        --itr 1 \
        --batch_size "$batch_size" \
        --learning_rate "$learning_rate" \
        --random_seed 2024 \
        --checkpoints "$checkpoints"

      completed=$((completed + 1))
      echo "FINISHED $completed/36 experiments"

    done
  done
done

echo "All $completed CycleNet/MLP experiments completed."
