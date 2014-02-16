#!/bin/bash

dataset_path=../../data/txt/tng-noheader-nowalls-nonalpha
bash qsub_tng3.sh $dataset_path 6-rs10k-rss10 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=400M,h_rt=4:00:00" \
    "--reservoir_size=10000 --init_num_docs=100 --rejuv_sample_size=10 --init_num_iters=2000"
bash qsub_tng3.sh $dataset_path 6-rs10k-rss30 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=400M,h_rt=4:00:00" \
    "--reservoir_size=10000 --init_num_docs=100 --rejuv_sample_size=30 --init_num_iters=2000"
bash qsub_tng3.sh $dataset_path 6-rs10k-rss100 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=400M,h_rt=4:00:00" \
    "--reservoir_size=10000 --init_num_docs=100 --rejuv_sample_size=100 --init_num_iters=2000"
bash qsub_tng3.sh $dataset_path 6-rs10k-rss300 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=400M,h_rt=4:00:00" \
    "--reservoir_size=10000 --init_num_docs=100 --rejuv_sample_size=300 --init_num_iters=2000"
bash qsub_tng3.sh $dataset_path 6-rs10k-rss1k \
    "-q text.q -tc 2 -l num_proc=1,mem_free=400M,h_rt=4:00:00" \
    "--reservoir_size=10000 --init_num_docs=100 --rejuv_sample_size=1000 --init_num_iters=2000"
