#!/bin/bash
source /root/miniconda3/bin/activate /root/miniconda3/envs/gar
python train_patch.py --dataset 'NBA' \
 --data_path '/opt/data/private/dataset/' \
 --batch 2 \
 --test_batch 2 \
 --depth 2 \
 --num_heads 4 \
 --top_k1 12\
 --top_k2 6\
 --num_frame 18 \
 --image_width 1280 \
 --image_height 720 \
 --lr 1e-6 \
 --weight_decay 1e-4 \
 --num_total_frame 72 \
 --num_activities 9 \
 --epochs 30 \
 --device "0" \
