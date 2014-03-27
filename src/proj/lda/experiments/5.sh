#!/bin/bash

dataset_path=../../../data/txt/tng-noheader-nowalls-nonalpha
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 5-ess5 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --ess-threshold=5.0 --init_num_iters=2000"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 5-ess10 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --ess-threshold=10.0 --init_num_iters=2000"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 5-ess20 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --ess-threshold=20.0 --init_num_iters=2000"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 5-ess40 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=2:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --ess-threshold=40.0 --init_num_iters=2000"
