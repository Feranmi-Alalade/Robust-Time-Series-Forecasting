#!/bin/bash
set -euo pipefail

cd /home/alalade/CycleNet

mkdir -p logs/solar_base_lookback

for PRED_LEN in 96 192
do
    for SEQ_LEN in 96 192 336 720
    do

        echo "=========================================="
        echo "Dataset: Solar"
        echo "Model: Base CycleNet"
        echo "Forecast horizon: $PRED_LEN"
        echo "Lookback window: $SEQ_LEN"
        echo "Random seed: 2024"
        echo "=========================================="

        python -u run.py \
          --is_training 1 \
          --root_path ./dataset/ \
          --data_path solar_AL.txt \
          --model_id Solar_Base_L${SEQ_LEN}_H${PRED_LEN} \
          --model CycleNet \
          --data Solar \
          --features M \
          --seq_len $SEQ_LEN \
          --pred_len $PRED_LEN \
          --enc_in 137 \
          --cycle 144 \
          --model_type mlp \
          --d_model 512 \
          --use_revin 0 \
          --train_epochs 30 \
          --patience 5 \
          --batch_size 64 \
          --learning_rate 0.002 \
          --random_seed 2024 \
          --itr 1 \
          2>&1 | tee logs/solar_base_lookback/solar_sl${SEQ_LEN}_pl${PRED_LEN}.log

    done
done

echo "All Solar Base CycleNet lookback experiments completed."