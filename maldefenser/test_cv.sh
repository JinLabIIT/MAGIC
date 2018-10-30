#!/bin/bash

DATA="${1-SMALLACFG}"
GPU="${2-1}"  # select the GPU number, 0-3

# general/default settings
gm=DGCNN  # model
gpu_or_cpu=gpu
mlp_type=vanilla # rap or vanilla
cache_path=cached_${DATA,,}_graphs
hp_path=test_cv.hp
use_cached_data=False
train_dir=data/SMALLACFG

CUDA_VISIBLE_DEVICES=${GPU} python3.7 cross_valid.py        \
  -seed 1                                                   \
  -data ${DATA}                                             \
  -train_dir ${train_dir}                                   \
  -gm $gm                                                   \
  -mode ${gpu_or_cpu}                                       \
  -mlp_type ${mlp_type}                                     \
  -use_cached_data ${use_cached_data}                       \
  -cache_path ${cache_path}                                 \
  -hp_path ${hp_path}

echo "Cross validatation history: ${DATA}Run0.csv"
cat ${DATA}Run0.csv
