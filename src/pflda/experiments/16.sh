#!/bin/bash

dataset_path=../../../data/txt/tng-noheader-nowalls-nonalpha
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 16-rs1k-b0.001 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=400M,h_rt=2:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --init_num_iters=2000 --beta=0.001"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 16-rs1k-b0.01 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=400M,h_rt=2:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --init_num_iters=2000 --beta=0.01"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 16-rs1k-b0.1 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=400M,h_rt=2:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --init_num_iters=2000 --beta=0.1"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 16-rs1k-b1 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=400M,h_rt=2:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --init_num_iters=2000 --beta=1"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 16-rs1k-b10 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=400M,h_rt=2:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --init_num_iters=2000 --beta=10"
