python validate.py --dataset 'volleyball' \
 --data_path '/opt/data/private/dataset/' \
 --batch 2 \
 --test_batch 2 \
 --depth 2 \
 --num_heads 4 \
 --top_k1 1 \
 --top_k2 4 \
 --num_frame 5 \
 --lr 1e-6 \
 --weight_decay 1e-4 \
 --num_total_frame 10 \
 --num_activities 8 \
 --model_path '/opt/data/private/code/DFWSGAR-master1/result/[volleyball]_DFGAR_<2023-10-11_03-47-06>_(base_seed1)/epoch8_86.24%.pth' \
 