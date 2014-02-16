#!/bin/bash

dataset_path=../../data/txt/tng-noheader-nowalls-nonalpha

bash qsub.sh $dataset_path tng 14-rs0 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=2G,h_rt=24:00:00" \
    "--reservoir_size=0 --init_num_docs=1000 --init_num_iters=2000 --num_topics=20"
bash qsub.sh $dataset_path tng 14-rs10 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=2G,h_rt=24:00:00" \
    "--reservoir_size=10 --init_num_docs=1000 --init_num_iters=2000 --num_topics=20"
bash qsub.sh $dataset_path tng 14-rs100 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=2G,h_rt=24:00:00" \
    "--reservoir_size=100 --init_num_docs=1000 --init_num_iters=2000 --num_topics=20"
bash qsub.sh $dataset_path tng 14-rs1k \
    "-q text.q -tc 2 -l num_proc=1,mem_free=2G,h_rt=24:00:00" \
    "--reservoir_size=1000 --init_num_docs=1000 --init_num_iters=2000 --num_topics=20"
bash qsub.sh $dataset_path tng 14-rs10k \
    "-q text.q -tc 2 -l num_proc=1,mem_free=2G,h_rt=24:00:00" \
    "--reservoir_size=10000 --init_num_docs=1000 --init_num_iters=2000 --num_topics=20"
