#!/bin/bash
python train_patch.py --dataset 'volleyball' \
 --data_path '/opt/data/private/dataset/' \
 --batch 6\
 --test_batch 6 \
 --depth 2 \
 --num_heads 2 \
 --top_k1 1 \
 --top_k2 4 \
 --lr 1e-6 \
 --image_width 1280 \
 --image_height 720 \
 --weight_decay 1e-4 \
 --num_frame 5 \
 --num_total_frame 10 \
 --epochs 30 \
 --num_activities 8 \
 --device "0" \