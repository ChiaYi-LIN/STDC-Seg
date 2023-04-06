export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch \
--nproc_per_node=2 train_text.py \
--respath checkpoints/train_STDC1-Seg/ \
--backbone STDCNet813 \
--mode train \
--n_img_per_gpu 12 \
--n_workers_train 2 \
--n_workers_val 1 \
--max_iter 60000 \
--use_boundary_8 True \
--pretrain_path checkpoints/STDCNet813M_73.91.tar