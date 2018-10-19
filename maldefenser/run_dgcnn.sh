#!/bin/bash

# input arguments
DATA="${1-SMALLACFG}"

# general settings
gm=DGCNN  # model
gpu_or_cpu=gpu
GPU=1  # select the GPU number, 0-3
mlp_type=vanilla # rap or vanilla
cache_file=cached_msacfg_graphs.pkl

# dataset-specific settings
case ${DATA} in
MSACFG)
  bsize=100
  num_epochs=640
  use_cached_data=True
  cache_file=cached_msacfg_graphs.pkl
  ;;
SMALLACFG)
  num_epochs=4
  bsize=4
  use_cached_data=False
  ;;
*)
  num_epochs=6
  use_cached_data=False
  ;;
esac

CUDA_VISIBLE_DEVICES=${GPU} python3.7 cross_valid.py \
  -seed 1 \
  -data ${DATA} \
  -gm $gm \
  -mode ${gpu_or_cpu} \
  -mlp_type ${mlp_type} \
  -use_cached_data ${use_cached_data} \
  -cache_file ${cache_file}
