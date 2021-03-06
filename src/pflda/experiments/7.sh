#!/bin/bash

dataset_path=../../../data/txt/tng-noheader-nowalls-nonalpha
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 7 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=2:00:00" \
    "--resample_propagate=1 --init_num_docs=100 --init_num_iters=2000"
