#!/bin/bash
work_path=$(dirname $0)
name=$(basename $work_path)
da=clip_data

# N threads according to your GPU
SEND_THREAD_NUM=2

###########################

tmp_fifofile="/tmp/$$.fifo"
mkfifo "$tmp_fifofile"
exec 6<>"$tmp_fifofile"
for ((i=0;i<$SEND_THREAD_NUM;i++));do
                 echo                                                                                    
done >&6


for p in `cat  data/${da}/all.list`
do 
    read -u6
    {
    id=${p}_PrismNet_pu
    ff=$work_path/out/evals/${id}.metrics
    lg=$work_path/out/log/${id}.log
    if [ ! -f $ff ] ; then 
        echo ${p}" ==="
        $srun $work_path/train.sh $p $da > $lg 2> $lg
    fi
    sleep 1
    echo >&6
    } &
    pid=$!
    echo $pid
done

wait
exec 6>&-
exit 0

