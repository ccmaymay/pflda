#!/bin/bash

dataset_path=../../../data/txt/tng-noheader-nowalls-nonalpha
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 14 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=400M,h_rt=4:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --init_num_iters=2000 --ltr_eval=t"
