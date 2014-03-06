#!/bin/bash

dataset_path=../../data/txt/tng-noheader-nowalls-nonalpha
mkdir -p 26/tng
i=1
printenv > "26/tng/${i}.log"
nohup bash run_gibbs.sh $dataset_path tng \
    "--init_num_iters=2000 --num_topics=20" \
    >> "26/tng/${i}.log" 2>&1 &
