
# CEUTrack

it consist of the following folds:

experiments: the config files about training and testing.
lib: details about model and training

scripts: many scripts for dataset preprocess.

tracking: tracking and evaluation scripts.


## train
    export CUDA_VISIBLE_DEVICES=6
    python tracking/train.py --script ceutrack --config ceutrack_coesot  \
    --save_dir ./output --mode multiple --nproc_per_node 1 --use_wandb  0
    python tracking/test.py   ceutrack ceutrack_coesot --dataset coesot --threads 4 --num_gpus 1
    python tracking/analysis_results.py --dataset coesot  --parameter_name ceutrack_coesot
