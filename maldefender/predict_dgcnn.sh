#!/bin/bash

DATA="${1-MSACFG}"
GPU="${2-1}"  # select the GPU number, 0-3
HP_PATH="${3-msacfg.hp}"

# general/default settings
gpu_or_cpu=gpu
train_dir=../../${DATA}/TrainSet
test_dir=../../${DATA}/TestSet
use_cached_data=True
cache_path=cached_${DATA,,}_graphs
norm_path=norm_${DATA,,}

CUDA_VISIBLE_DEVICES=${GPU} python3.7 tuned_model.py        \
  -seed 1                                                   \
  -data ${DATA}                                             \
  -train_dir ${train_dir}                                   \
  -test_dir ${test_dir}                                     \
  -mode ${gpu_or_cpu}                                       \
  -gpu_id ${GPU}                                            \
  -use_cached_data ${use_cached_data}                       \
  -cache_path ${cache_path}                                 \
  -norm_path ${norm_path}                                   \
  -hp_path ${HP_PATH}

echo "Tuned model prediction from ${test_dir}/submission.csv"
head -n10 ${test_dir}/submission.csv
