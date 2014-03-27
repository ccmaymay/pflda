#!/bin/bash

dataset_path=../../../data/txt/tng-noheader-nowalls-nonalpha
dataset_name=diff3
init_seed=980044224
init_num_docs=200
init_num_iters=200

bash run_pf_qsub_wrapper.sh $dataset_path $dataset_name 28-rs50000-rss5000 \
    "-q text.q -t 1-30 -tc 4 -l num_proc=1,mem_free=2G,h_rt=48:00:00" \
    "--reservoir_size=50000 --rejuv_sample_size=5000 --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --init_seed=$init_seed"
bash run_pf_qsub_wrapper.sh $dataset_path $dataset_name 28-rs100000-rss10000 \
    "-q text.q -t 1-30 -tc 4 -l num_proc=1,mem_free=2G,h_rt=48:00:00" \
    "--reservoir_size=100000 --rejuv_sample_size=10000 --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --init_seed=$init_seed"
bash run_pf_qsub_wrapper.sh $dataset_path $dataset_name 28-rs500000-rss50000 \
    "-q text.q -t 1-30 -tc 4 -l num_proc=1,mem_free=4G,h_rt=48:00:00" \
    "--reservoir_size=500000 --rejuv_sample_size=50000 --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --init_seed=$init_seed"
