#!/bin/bash

dataset_path=../../../data/txt/tng-noheader-nowalls-nonalpha
for dataset_name in diff3 sim3 rel3
do
    if [ $dataset_name == diff3 ]
    then
        init_num_docs=167
        init_num_iters=200
    elif [ $dataset_name == sim3 ]
    then
        init_num_docs=177
        init_num_iters=200
    elif [ $dataset_name == rel3 ]
    then
        init_num_docs=158
        init_num_iters=200
    fi

    bash run_pf_qsub_wrapper.sh $dataset_path $dataset_name 27-rs0 \
        "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=2:00:00" \
        "--reservoir_size=0 --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters" \
    bash run_pf_qsub_wrapper.sh $dataset_path $dataset_name 27-rs1k \
        "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=400M,h_rt=4:00:00" \
        "--reservoir_size=1000 --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters" \
done
