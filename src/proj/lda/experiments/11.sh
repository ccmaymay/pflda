#!/bin/bash

dataset_path=../../../data/txt/twitter
bash qsub.sh $dataset_path null 11-t3 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=1G,h_rt=12:00:00" \
    "--reservoir_size=1000 --init_num_docs=500 --num_topics=3 --init_num_iters=2000"
bash qsub.sh $dataset_path null 11-t6 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=1G,h_rt=12:00:00" \
    "--reservoir_size=1000 --init_num_docs=500 --num_topics=6 --init_num_iters=2000"
bash qsub.sh $dataset_path null 11-t12 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=1G,h_rt=12:00:00" \
    "--reservoir_size=1000 --init_num_docs=500 --num_topics=12 --init_num_iters=2000"
