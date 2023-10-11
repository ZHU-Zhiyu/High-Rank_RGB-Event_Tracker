CUDA_VISIBLE_DEVICES=2,3,4,5 python tracking/train.py --script ceutrack --config ceutrack_coesot  \
    --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb  0 >> log.txt