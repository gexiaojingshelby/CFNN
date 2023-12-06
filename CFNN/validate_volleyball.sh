source /root/miniconda3/bin/activate /root/miniconda3/envs/gar
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
 --epochs 45 \
 --intro 'res18_weight_spa_channel_clstoken_clspatchc_lr6' \
 --model_path '/opt/data/private/code/DFWSGAR-master1/result/[volleyball]_DFGAR_<2023-10-11_03-47-06>_(base_seed1)/epoch8_86.24%.pth' \
 #  --draw_dir './output_all/vvv'\
 #  --model_path '/opt/data/share/104011/DFWSGAR-master1/result/[volleyball]_DFGAR_<2023-02-02_16-20-39>_(lr16_wd4_normal_backbone)/epoch24_89.23%.pth' \