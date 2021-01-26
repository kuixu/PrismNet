#!/bin/bash
d=clip_data
for p in `cat data/${d}/all.list`
do 
    python -u tools/generate_dataset.py $p 1 5 data/$d
done
