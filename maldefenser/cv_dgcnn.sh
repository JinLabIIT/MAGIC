#!/bin/bash

DATA="${1-MSACFG}"
GPU="${2-1}"  # select the GPU number, 0-3

# general/default settings
gm=DGCNN  # model
gpu_or_cpu=gpu
mlp_type=vanilla # rap or vanilla
cache_path=cached_${DATA,,}_graphs
hp_path=train_once.hp
train_dir=../TrainSet
test_dir=../TestSet
use_cached_data=True

CUDA_VISIBLE_DEVICES=${GPU} python3.7 cross_valid.py        \
  -seed 1                                                   \
  -data ${DATA}                                             \
  -train_dir ${train_dir}                                   \
  -test_dir ${test_dir}                                     \
  -gm $gm                                                   \
  -mode ${gpu_or_cpu}                                       \
  -gpu_id ${GPU}                                            \
  -mlp_type ${mlp_type}                                     \
  -use_cached_data ${use_cached_data}                       \
  -cache_path ${cache_path}                                 \
  -hp_path ${hp_path}

echo "Cross validatation history:"
head -n10 ${DATA}Run0.csv
