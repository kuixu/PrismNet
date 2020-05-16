#!/bin/bash
work_path=$(dirname $0)
name=$(basename $work_path)
echo `date +%Y%m%d%H%M%S`

p=${1}

exp=$name

out_dir=$work_path/out/$p/$ss_type

python -u tools/train.py \
    --datadir data \
    --out_dir $out_dir \
    --batch_size 64 \
    --log_interval 100\
    --exp_name $exp\
    --p_name $p\
    --beta 2 \
    --eval \
