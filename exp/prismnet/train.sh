#!/bin/bash
work_path=$(dirname $0)
name=$(basename $work_path)
echo `date +%Y%m%d%H%M%S`

p=${1}

exp=$name

out_dir=$work_path/out/$p/$ss_type
mkdir -p $out_dir
python -u tools/train.py \
    --datadir data \
    --out_dir $out_dir \
    --batch_size 64 \
    --log_interval 100\
    --exp_name $exp\
    --p_name $p\
    --lr 0.001 \
    --beta 2 \
    --lr_scheduler "warmup" \
    --weight_decay 0.000001\
    --nepochs 200 \
    --seed 1024 \
    --tfboard \
    | tee $out_dir/log.txt
