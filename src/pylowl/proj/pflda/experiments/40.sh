#!/bin/bash

# single particle, no init, new dataset path, 1k reservoir, larger rejuvenation samples

dataset_path=../../../../data/txt/tng-noheader-nowalls-nonalpha
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 40-rss20 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--num_particles=1 --ess_threshold=2 --reservoir_size=1000 --init_num_docs=0 --init_num_iters=0 --rejuv_sample_size=20"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 40-rss30 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--num_particles=1 --ess_threshold=2 --reservoir_size=1000 --init_num_docs=0 --init_num_iters=0 --rejuv_sample_size=30"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 40-rss100 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--num_particles=1 --ess_threshold=2 --reservoir_size=1000 --init_num_docs=0 --init_num_iters=0 --rejuv_sample_size=100"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 40-rss300 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--num_particles=1 --ess_threshold=2 --reservoir_size=1000 --init_num_docs=0 --init_num_iters=0 --rejuv_sample_size=300"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 40-rss1000 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--num_particles=1 --ess_threshold=2 --reservoir_size=1000 --init_num_docs=0 --init_num_iters=0 --rejuv_sample_size=1000"
