#!/bin/bash

# no init, new dataset path, full reservoir

dataset_path=../../../../data/txt/tng-noheader-nowalls-nonalpha
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 41-rss20 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--reservoir_size=300000 --init_num_docs=0 --init_num_iters=0 --rejuv_sample_size=20"
bash run_pf_qsub_wrapper_tng3.sh $dataset_path 41-rss30 \
    "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--reservoir_size=300000 --init_num_docs=0 --init_num_iters=0 --rejuv_sample_size=30"
