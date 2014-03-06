#!/bin/bash

dataset_path=../../data/txt/twitter
bash qsub.sh $dataset_path null 13-rs0 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=1G,h_rt=12:00:00" \
    "--reservoir_size=0 --init_num_docs=500 --init_num_iters=2000 --num_topics=6"
bash qsub.sh $dataset_path null 13-rs10 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=1G,h_rt=12:00:00" \
    "--reservoir_size=10 --init_num_docs=500 --init_num_iters=2000 --num_topics=6"
bash qsub.sh $dataset_path null 13-rs100 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=1G,h_rt=12:00:00" \
    "--reservoir_size=100 --init_num_docs=500 --init_num_iters=2000 --num_topics=6"
bash qsub.sh $dataset_path null 13-rs1k \
    "-q text.q -tc 2 -l num_proc=1,mem_free=1G,h_rt=12:00:00" \
    "--reservoir_size=1000 --init_num_docs=500 --init_num_iters=2000 --num_topics=6"
bash qsub.sh $dataset_path null 13-rs10k \
    "-q text.q -tc 2 -l num_proc=1,mem_free=2G,h_rt=24:00:00" \
    "--reservoir_size=10000 --init_num_docs=500 --init_num_iters=2000 --num_topics=6"
