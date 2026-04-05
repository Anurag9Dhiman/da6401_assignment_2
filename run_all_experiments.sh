#!/bin/bash
# run_all_experiments.sh
# This script sequentially runs all the required training and analysis jobs.
# Warning: This will take several hours or a full day depending on your hardware!

# Fail on error
set -e

echo "Starting DA6401 Assignment 2 Full Pipeline Execution..."

echo "============================================="
echo "Step 1: Download and Base Classification"
echo "============================================="
python3 train.py --download --task cls 

echo "============================================="
echo "Step 2: Classification Runs (BatchNorm & Dropout variations)"
echo "============================================="
# 2.1 — with BatchNorm (default)
python3 train.py --task cls --dropout_p 0.5 --run_name bn_on --epochs 30

# 2.1 — without BatchNorm
python3 train.py --task cls --dropout_p 0.5 --no-batch_norm --run_name bn_off --epochs 30

# 2.2 — no dropout
python3 train.py --task cls --dropout_p 0.0 --run_name no_dropout --epochs 30

# 2.2 — dropout 0.2
python3 train.py --task cls --dropout_p 0.2 --run_name dropout_0.2 --epochs 30

echo "============================================="
echo "Step 3: Localization"
echo "============================================="
python3 train.py --task loc --pretrained_cls checkpoints/vgg11_cls.pth --freeze_encoder none --run_name loc --epochs 30

echo "============================================="
echo "Step 4: Segmentation (Transfer Learning Showdown)"
echo "============================================="
python3 train.py --task seg --pretrained_cls checkpoints/vgg11_cls.pth --freeze_encoder full --run_name seg_frozen --epochs 30

python3 train.py --task seg --pretrained_cls checkpoints/vgg11_cls.pth --freeze_encoder partial --run_name seg_partial --epochs 30

python3 train.py --task seg --pretrained_cls checkpoints/vgg11_cls.pth --freeze_encoder none --run_name seg_full --epochs 30

echo "============================================="
echo "Step 5: Multi-task"
echo "============================================="
python3 train.py --task multi --pretrained_cls checkpoints/vgg11_cls.pth --run_name multi --epochs 30

echo "============================================="
echo "Step 6: Analysis Scripts"
echo "============================================="
# Note: You may need to change 'path/to/dog.jpg' and 'petX.jpg' to actual paths on your machine.
python3 analyze.py --checkpoint checkpoints/vgg11_cls.pth --image data/pets/images/Abyssinian_1.jpg
python3 evaluate_loc.py --checkpoint checkpoints/localization.pth

echo "Pipeline execution completed!"
