export CUDA_VISIBLE_DEVICES=2,3,4,5
python -m torch.distributed.launch \
--nproc_per_node=4 train_text.py \
--respath checkpoints/train_STDC1-CSCText/ \
--backbone STDCNet813 \
--mode train \
--n_img_per_gpu 6 \
--n_workers_train 12 \
--n_workers_val 1 \
--max_iter 160000 \
--save_iter_sep 10000 \
--use_boundary_8 True \
--pretrain_path checkpoints/STDCNet813M_73.91.tar