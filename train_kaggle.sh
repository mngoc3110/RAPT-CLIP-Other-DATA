#!/bin/bash

# [KAGGLE TRAINING SCRIPT]
# Automatically prepares balanced labels in /kaggle/working and runs training.

# 1. Define Paths
KAGGLE_INPUT_ROOT="/kaggle/input/datasets/mngochocsupham/daisee/DAiSEE_data"
WORKING_DIR="/kaggle/working"
LABEL_OUT_DIR="$WORKING_DIR/labels"

# Create output directory for labels
mkdir -p "$LABEL_OUT_DIR"

echo "================================================================"
echo "STEP 1: Generating Balanced Labels in $LABEL_OUT_DIR"
echo "================================================================"

# Run the balancing script pointing to read-only input and writable output
python balance_daisee.py \
  --train-in "$KAGGLE_INPUT_ROOT/Labels/TrainLabels.csv" \
  --val-in "$KAGGLE_INPUT_ROOT/Labels/ValidationLabels.csv" \
  --test-in "$KAGGLE_INPUT_ROOT/Labels/TestLabels.csv" \
  --train-out "$LABEL_OUT_DIR/TrainLabels_Balanced.csv" \
  --val-out "$LABEL_OUT_DIR/ValidationLabels_Balanced.csv" \
  --test-out "$LABEL_OUT_DIR/TestLabels_Balanced.csv"

echo "================================================================"
echo "STEP 2: Starting Training"
echo "================================================================"

# PyTorch Optimization for CUDA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run Main Training Loop
# Note:
# --root-dir points to the Video Dataset (Read-only Input)
# --*-annotation points to the Balanced Labels (Writable Working Dir)
# --gpu 0 (Kaggle usually provides P100/T4 on cuda:0)

python main.py \
  --mode train \
  --exper-name Train_DAISEE_Kaggle \
  --dataset DAISEE \
  --gpu 0 \
  --epochs 20 \
  --batch-size 4 \
  --resume /kaggle/input/resume-rapt-clip-daisee/model.pth \
  --accumulation-steps 1 \
  --workers 4 \
  --optimizer AdamW \
  --lr 2e-5 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 2e-4 \
  --lr-adapter 1e-4 \
  --weight-decay 0.0005 \
  --milestones 10 20 \
  --gamma 0.1 \
  --temporal-layers 1 \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 50 \
  --root-dir "$KAGGLE_INPUT_ROOT/DataSet" \
  --train-annotation "$LABEL_OUT_DIR/TrainLabels_Balanced.csv" \
  --val-annotation "$LABEL_OUT_DIR/ValidationLabels_Balanced.csv" \
  --test-annotation "$LABEL_OUT_DIR/TestLabels_Balanced.csv" \
  --clip-path ViT-B/16 \
  --text-type prompt_ensemble \
  --temporal-type attn_pool \
  --use-adapter True \
  --contexts-number 8 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  --lambda_dc 0.1 \
  --dc-warmup 5 \
  --dc-ramp 10 \
  --lambda_mi 0.1 \
  --mi-warmup 5 \
  --mi-ramp 10 \
  --temperature 0.07 \
  --use-ldl \
  --ldl-temperature 1.0 \
  --use-amp \
  --grad-clip 1.0 \
  --mixup-alpha 0.2
