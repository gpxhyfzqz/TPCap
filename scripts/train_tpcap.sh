CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 ./train_tpcap.py \
    --out_dir results/train_tpcap