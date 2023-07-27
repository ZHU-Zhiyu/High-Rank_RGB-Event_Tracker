CUDA_VISIBLE_DEVICES=7 python tracking/train.py --script ceutrack --config ceutrack_coesot  \
    --save_dir ./output --mode multiple --nproc_per_node 1 --use_wandb  0