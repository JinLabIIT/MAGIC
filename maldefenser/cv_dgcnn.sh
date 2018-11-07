#!/bin/bash

DATA="${1-MSACFG}"
GPU="${2-1}"  # select the GPU number, 0-3

# general/default settings
gpu_or_cpu=gpu
cache_path=cached_${DATA,,}_graphs
# hp_path=full_gpu${GPU}.hp
hp_path=train_once.hp
train_dir=../TrainSet
test_dir=../TestSet
use_cached_data=True

CUDA_VISIBLE_DEVICES=${GPU} python3.7 cross_valid.py        \
  -seed 1                                                   \
  -data ${DATA}                                             \
  -train_dir ${train_dir}                                   \
  -test_dir ${test_dir}                                     \
  -mode ${gpu_or_cpu}                                       \
  -gpu_id ${GPU}                                            \
  -use_cached_data ${use_cached_data}                       \
  -cache_path ${cache_path}                                 \
  -hp_path ${hp_path}

echo "Cross validatation history:"
head -n10 ${DATA}Gpu${GPU}Run0.csv
