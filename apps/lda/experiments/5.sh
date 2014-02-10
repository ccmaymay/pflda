#!/bin/bash

dataset_path=../../data/txt/tng-nonalpha
bash qsub_tng3.sh $dataset_path 5-ess5 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --ess-threshold=5.0"
bash qsub_tng3.sh $dataset_path 5-ess10 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --ess-threshold=10.0"
bash qsub_tng3.sh $dataset_path 5-ess20 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --ess-threshold=20.0"
bash qsub_tng3.sh $dataset_path 5-ess40 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=200M,h_rt=2:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --ess-threshold=40.0"
