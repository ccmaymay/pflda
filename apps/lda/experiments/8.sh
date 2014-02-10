#!/bin/bash

dataset_path=../../data/txt/tng-nonalpha
bash qsub_tng3.sh $dataset_path 8-t2 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --num_topics=2 --init_num_iters=2000"
bash qsub_tng3.sh $dataset_path 8-t3 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --num_topics=3 --init_num_iters=2000"
bash qsub_tng3.sh $dataset_path 8-t4 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --num_topics=4 --init_num_iters=2000"
bash qsub_tng3.sh $dataset_path 8-t5 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --num_topics=5 --init_num_iters=2000"
bash qsub_tng3.sh $dataset_path 8-t6 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --num_topics=6 --init_num_iters=2000"
