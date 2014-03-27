#!/bin/bash

dataset_path=../../../data/txt/tng-noheader-nowalls-nonalpha
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 15-rs1k-a0.001 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=400M,h_rt=2:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --init_num_iters=2000 --alpha=0.001"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 15-rs1k-a0.01 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=400M,h_rt=2:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --init_num_iters=2000 --alpha=0.01"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 15-rs1k-a0.1 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=400M,h_rt=2:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --init_num_iters=2000 --alpha=0.1"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 15-rs1k-a1 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=400M,h_rt=2:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --init_num_iters=2000 --alpha=1"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 15-rs1k-a10 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=400M,h_rt=2:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --init_num_iters=2000 --alpha=10"
