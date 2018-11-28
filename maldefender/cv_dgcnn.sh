#!/bin/bash

DATA="${1-MSACFG}"
GPU="${2-1}"  # select the GPU number, 0-3
HP_PATH="${3-msacfg.hp}" # full_gpu${GPU}.hp

# general/default settings
gpu_or_cpu=gpu
train_dir=../../${DATA}/TrainSet
use_cached_data=True
cache_path=cached_${DATA,,}_graphs
norm_path=norm_${DATA,,}

CUDA_VISIBLE_DEVICES=${GPU} python3.7 cross_valid.py        \
  -seed 1                                                   \
  -data ${DATA}                                             \
  -train_dir ${train_dir}                                   \
  -mode ${gpu_or_cpu}                                       \
  -gpu_id ${GPU}                                            \
  -use_cached_data ${use_cached_data}                       \
  -cache_path ${cache_path}                                 \
  -norm_path ${norm_path}                                   \
  -hp_path ${HP_PATH}

echo "Cross validatation history:"
head -n10 ${DATA}Gpu${GPU}Run0.csv
