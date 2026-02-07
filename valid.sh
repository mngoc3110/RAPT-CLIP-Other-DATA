#!/bin/bash

# Script đánh giá cho bộ dữ liệu DAISEE.

python main.py \
    --mode eval \
    --dataset DAISEE \
    --gpu 0 \
    --exper-name eval_daisee \
    --eval-checkpoint outputs/Train_DAISEE-[...]/model_best.pth \
    --root-dir ./DAISEE/DataSet \
    --train-annotation ./DAISEE/Labels/TrainLabels.csv \
    --val-annotation ./DAISEE/Labels/ValidationLabels.csv \
    --test-annotation ./DAISEE/Labels/TestLabels.csv \
    --clip-path ViT-B/16 \
    --text-type prompt_ensemble \
    --contexts-number 8 \
    --class-token-position end \
    --class-specific-contexts True \
    --load_and_tune_prompt_learner True \
    --temporal-layers 1 \
    --num-segments 16 \
    --duration 1 \
    --image-size 224 \
    --seed 42 \
    --temperature 0.07 \
    --use-amp
