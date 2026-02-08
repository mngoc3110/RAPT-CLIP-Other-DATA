#!/bin/bash

# [LUỒNG 1: KHỞI ĐỘNG]
# Đây là file bắt đầu (Entry Point) được cấu hình cho DAISEE.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python main.py \
  --mode train \
  --exper-name Train_DAISEE \
  --dataset DAISEE \
  --gpu mps \
  --epochs 25 \
  --batch-size 4 \
  --accumulation-steps 1 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 1e-4 \
  --lr-adapter 5e-5 \
  --weight-decay 0.0005 \
  --milestones 10 20 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 10 \
  --root-dir ./DAISEE/DataSet \
  --train-annotation ./DAISEE/Labels/TrainLabels_Balanced.csv \
  --val-annotation ./DAISEE/Labels/ValidationLabels_Balanced.csv \
  --test-annotation ./DAISEE/Labels/TestLabels_Balanced.csv \
  --clip-path ViT-B/16 \
  --text-type prompt_ensemble \
  --temporal-type attn_pool \
  --use-adapter True \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --lambda_dc 0.05 \
  --dc-warmup 5 \
  --dc-ramp 15 \
  --lambda_mi 0.05 \
  --mi-warmup 5 \
  --mi-ramp 15 \
  --temperature 0.07 \
  --use-ldl \
  --ldl-temperature 1.0 \
  --use-amp \
  --grad-clip 1.0 \
  --mixup-alpha 0.2
