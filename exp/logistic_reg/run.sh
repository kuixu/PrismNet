#!/bin/bash
work_path=$(dirname $0)

p=HEK293_RBP_HL_bind_matrix_total
p2=$p
la=10

# part=Test
mkdir $work_path/out
mkdir $work_path/out/models
mkdir $work_path/out/log

train_data='data/halflife/'$p'.train.npz'
test_data='data/halflife/'$p'.test.npz'
pred_data='data/halflife/'${p2}'.test.npz'
# CUDA_VISIBLE_DEVICE="0" 
python -u exp/logistic_reg/main.py \
  --train_data $train_data \
  --test_data $test_data \
  --pred_data $pred_data \
  --model_path $work_path/out/models/${p}_best.model \
  --lam $la \
  ${@:4}| tee -a $work_path/out/log/${p}.txt 
