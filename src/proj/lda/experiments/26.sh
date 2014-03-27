#!/bin/bash

dataset_path=../../../data/txt/tng-noheader-nowalls-nonalpha
dataset_name=tng
bash qsub_gibbs.sh $dataset_path $dataset_name 26 \
    "-q text.q -t 1 -l num_proc=1,mem_free=2G,h_rt=24:00:00" \
    "--init_num_iters=2000 --num_topics=20"
