#!/bin/bash

dataset_path=../../data/txt/tng-noheader-nowalls-nonalpha
for dataset_name in diff3 sim3 rel3
do
    if [ $dataset_name == diff3 ]
    then
        init_num_docs=200
        init_num_iters=200
    elif [ $dataset_name == sim3 ]
    then
        init_num_docs=200
        init_num_iters=200
    elif [ $dataset_name == rel3 ]
    then
        init_num_docs=200
        init_num_iters=200
    fi

    nohup bash qsub.sh $dataset_path $dataset_name 23 \
        "-q text.q -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
        "--init_num_docs=$init_num_docs --init_num_iters=$init_num_iters" \
        >/dev/null 2>&1 &
done
