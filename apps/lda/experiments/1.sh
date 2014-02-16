#!/bin/bash

dataset_path=../../data/txt/tng-noheader-nowalls-nonalpha
for dataset_name in diff3 sim3 rel3
do
    if [ $dataset_name == diff3 ]
    then
        init_num_docs=167
        init_num_iters=2000
    elif [ $dataset_name == sim3 ]
    then
        init_num_docs=177
        init_num_iters=2000
    elif [ $dataset_name == rel3 ]
    then
        init_num_docs=158
        init_num_iters=2000
    fi

    bash qsub.sh $dataset_path $dataset_name 1-rs0 \
        "-q text.q -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
        "--reservoir_size=0 --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters"
    bash qsub.sh $dataset_path $dataset_name 1-rs10 \
        "-q text.q -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
        "--reservoir_size=10 --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters"
    bash qsub.sh $dataset_path $dataset_name 1-rs100 \
        "-q text.q -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
        "--reservoir_size=100 --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters"
    bash qsub.sh $dataset_path $dataset_name 1-rs1k \
        "-q text.q -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
        "--reservoir_size=1000 --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters"
    bash qsub.sh $dataset_path $dataset_name 1-rs10k \
        "-q text.q -tc 2 -l num_proc=1,mem_free=500M,h_rt=4:00:00" \
        "--reservoir_size=10000 --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters"
    bash qsub.sh $dataset_path $dataset_name 1-rs100k \
        "-q text.q -tc 2 -l num_proc=1,mem_free=2G,h_rt=12:00:00" \
        "--reservoir_size=100000 --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters"
    bash qsub.sh $dataset_path $dataset_name 1-rs500k \
        "-q text.q -tc 2 -l num_proc=1,mem_free=4G,h_rt=12:00:00" \
        "--reservoir_size=500000 --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters"
done
