#!/bin/bash

dataset_path=../../../data/txt/tng-noheader-nowalls-nonalpha
dataset_name=diff3
for alpha in 0.001 0.01 0.1 1 10
do
    for beta in 0.001 0.01 0.1 1 10
    do
        bash run_pf_qsub_wrapper.sh $dataset_path $dataset_name 29-a$alpha-b$beta \
            "-q text.q -t 1-30 -tc 1 -l num_proc=1,mem_free=400M,h_rt=2:00:00" \
            "--reservoir_size=1000 --init_num_docs=100 --init_num_iters=2000"
    done
done
