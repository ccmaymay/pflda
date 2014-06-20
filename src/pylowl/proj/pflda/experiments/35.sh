#!/bin/bash

# 2 with single particle, new dataset path

dataset_path=../../../../data/txt/tng-noheader-nowalls-nonalpha
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 2-rs1k-ibs0 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--num_particles=1 --ess_threshold=2 --reservoir_size=1000 --init_num_docs=0 --init_num_iters=0"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 2-rs1k-ibs10 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--num_particles=1 --ess_threshold=2 --reservoir_size=1000 --init_num_docs=10 --init_num_iters=2000"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 2-rs1k-ibs30 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--num_particles=1 --ess_threshold=2 --reservoir_size=1000 --init_num_docs=30 --init_num_iters=2000"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 2-rs1k-ibs100 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--num_particles=1 --ess_threshold=2 --reservoir_size=1000 --init_num_docs=100 --init_num_iters=2000"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 2-rs1k-ibs300 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--num_particles=1 --ess_threshold=2 --reservoir_size=1000 --init_num_docs=300 --init_num_iters=2000"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 2-rs1k-ibs1k \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--num_particles=1 --ess_threshold=2 --reservoir_size=1000 --init_num_docs=1000 --init_num_iters=2000"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 2-rs1k-ibs3k \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--num_particles=1 --ess_threshold=2 --reservoir_size=1000 --init_num_docs=3000 --init_num_iters=2000"
