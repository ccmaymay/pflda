#!/bin/bash

# single particle, no init, new dataset path, full reservoir, larger rejuvenation samples

dataset_path=../../../../data/txt/tng-noheader-nowalls-nonalpha
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 39-rss1 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--num_particles=1 --ess_threshold=2 --reservoir_size=300000 --init_num_docs=0 --init_num_iters=0 --rejuv_sample_size=1"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 39-rss3 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--num_particles=1 --ess_threshold=2 --reservoir_size=300000 --init_num_docs=0 --init_num_iters=0 --rejuv_sample_size=3"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 39-rss10 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--num_particles=1 --ess_threshold=2 --reservoir_size=300000 --init_num_docs=0 --init_num_iters=0 --rejuv_sample_size=10"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 39-rss20 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--num_particles=1 --ess_threshold=2 --reservoir_size=300000 --init_num_docs=0 --init_num_iters=0 --rejuv_sample_size=20"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 39-rss30 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--num_particles=1 --ess_threshold=2 --reservoir_size=300000 --init_num_docs=0 --init_num_iters=0 --rejuv_sample_size=30"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 39-rss100 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--num_particles=1 --ess_threshold=2 --reservoir_size=300000 --init_num_docs=0 --init_num_iters=0 --rejuv_sample_size=100"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 39-rss300 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--num_particles=1 --ess_threshold=2 --reservoir_size=300000 --init_num_docs=0 --init_num_iters=0 --rejuv_sample_size=300"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 39-rss1000 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--num_particles=1 --ess_threshold=2 --reservoir_size=300000 --init_num_docs=0 --init_num_iters=0 --rejuv_sample_size=1000"
