#!/bin/bash

# input arguments
DATA="${1-SMALLACFG}"
fold=${2-1}  # which fold as testing data
test_number=${3-0}  # if specified, use the last test_number graphs as test data

# general settings
gm=DGCNN  # model
gpu_or_cpu=gpu
GPU=1  # select the GPU number, 0-3
CONV_SIZE="32-32-32-1"
sortpooling_k=0.6  # If k <= 1, then k is set to an integer so that k% of graphs have nodes less than this integer
FP_LEN=0  # final dense layer's input dimension, decided by data
n_hidden=128  # final dense layer's hidden size
mlp_type=vanilla # rap or vanilla

bsize=40  # batch size
num_epochs=400
learning_rate=0.0001
dropout=True
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
  num_epochs=4
  use_cached_data=False
  ;;
esac

if [ ${fold} == 0 ]; then
  rm result.txt
  echo "Running 10-fold cross validation"
  start=`date +%s`
  for i in $(seq 1 10)
  do
    CUDA_VISIBLE_DEVICES=${GPU} python train_model.py \
        -seed 1 \
        -data ${DATA} \
        -fold ${i} \
        -learning_rate ${learning_rate} \
        -num_epochs ${num_epochs} \
        -hidden ${n_hidden} \
        -latent_dim ${CONV_SIZE} \
        -sortpooling_k ${sortpooling_k} \
        -out_dim ${FP_LEN} \
        -batch_size ${bsize} \
        -gm ${gm} \
        -mode ${gpu_or_cpu} \
        -mlp_type ${mlp_type} \
        -dropout ${dropout} \
        -use_cached_data ${use_cached_data} \
        -cache_file ${cache_file}
  done
  stop=`date +%s`
  echo "End of cross-validation"
  echo "The total running time is $[stop - start] seconds."
  echo "The accuracy results for ${DATA} are as follows:"
  cat result.txt
  echo "Average accuracy is"
  cat result.txt | awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }'
else
  CUDA_VISIBLE_DEVICES=${GPU} python train_model.py \
      -seed 1 \
      -data ${DATA} \
      -fold ${fold} \
      -learning_rate ${learning_rate} \
      -num_epochs ${num_epochs} \
      -hidden ${n_hidden} \
      -latent_dim ${CONV_SIZE} \
      -sortpooling_k ${sortpooling_k} \
      -out_dim ${FP_LEN} \
      -batch_size ${bsize} \
      -gm $gm \
      -mode ${gpu_or_cpu} \
      -mlp_type ${mlp_type} \
      -dropout ${dropout} \
      -use_cached_data ${use_cached_data} \
      -cache_file ${cache_file} \
      -test_number ${test_number}
  echo "Train confusion matrix:"
  cat ${DATA}_train_confusion_matrix.txt
  echo "Test confusion matrix:"
  cat ${DATA}_test_confusion_matrix.txt
fi
