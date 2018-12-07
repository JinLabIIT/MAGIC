#!/bin/bash

DATA="${1-MSACFG}"
GPU="${2-2}"  # select the GPU number, 0-3
HP_PATH="${3-msacfg.hp}"
ATTR_GROUP="${4-AttrGroup1}"
# general/default settings
gpu_or_cpu=gpu

# dataset with only attributes from instructions, not including n-grams
input_dir=../../${DATA}/${ATTR_GROUP}
train_dir=${input_dir}
test_dir=${input_dir}
use_cached_data=True
cache_path=${input_dir}/cached_${DATA,,}_graphs
norm_path=${input_dir}/norm_${DATA,,}
norm_op=none
# model_date=04-Dec-2018-08:16:52
# model_date=04-Dec-2018-14:12:39
model_date=04-Dec-2018-21:03:40

CUDA_VISIBLE_DEVICES=${GPU} python3.7 predict_model.py      \
  -seed 1                                                   \
  -data ${DATA}                                             \
  -train_dir ${train_dir}                                   \
  -test_dir ${test_dir}                                     \
  -mode ${gpu_or_cpu}                                       \
  -gpu_id ${GPU}                                            \
  -use_cached_data ${use_cached_data}                       \
  -cache_path ${cache_path}                                 \
  -norm_path ${norm_path}                                   \
  -norm_op ${norm_op}                                       \
  -model_date ${model_date}                                 \
  -hp_path ${HP_PATH}

echo "Model prediction saved to ${test_dir}/submission.csv"
head -n10 ${test_dir}/submission.csv
kaggle competitions submit -c malware-classification -f ${test_dir}/submission.csv -m "prediction by model(${model_date})"
