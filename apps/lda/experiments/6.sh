#!/bin/bash

bash qsub_tng3.sh 6-rs10000-rss10 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=400M,h_rt=2:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --rejuv_sample_size=10"
bash qsub_tng3.sh 6-rs10000-rss30 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=400M,h_rt=2:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --rejuv_sample_size=30"
bash qsub_tng3.sh 6-rs10000-rss100 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=400M,h_rt=2:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --rejuv_sample_size=100"
bash qsub_tng3.sh 6-rs10000-rss300 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=400M,h_rt=2:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --rejuv_sample_size=300"
bash qsub_tng3.sh 6-rs10000-rss1000 \
    "-q text.q -tc 2 -l num_proc=1,mem_free=400M,h_rt=2:00:00" \
    "--reservoir_size=1000 --init_num_docs=100 --rejuv_sample_size=1000"
