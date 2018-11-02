#!/bin/bash

DATA="${1-MSACFG}"
GPU="${2-1}"  # select the GPU number, 0-3

# general/default settings
gm=DGCNN  # model
gpu_or_cpu=gpu
mlp_type=logistic_reg # rap or vanilla
cache_path=cached_${DATA,,}_graphs
train_dir=../TrainSet
test_dir=../TestSet
use_cached_data=True

CUDA_VISIBLE_DEVICES=${GPU} python3.7 tuned_model.py        \
  -seed 1                                                   \
  -data ${DATA}                                             \
  -train_dir ${train_dir}                                   \
  -test_dir ${test_dir}                                     \
  -gm $gm                                                   \
  -mode ${gpu_or_cpu}                                       \
  -mlp_type ${mlp_type}                                     \
  -use_cached_data ${use_cached_data}                       \
  -cache_path ${cache_path}

echo "Tuned model prediction from ${test_dir}/submission.csv"
head -n10 ${test_dir}/submission.csv
