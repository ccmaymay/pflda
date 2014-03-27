#!/bin/bash

init_num_docs=1000000
init_num_iters=10000

bash run_gibbs_qsub_wrapper.sh ../../../data/txt/twitter null 31-t5 \
    "-q text.q -t 1-5 -tc 2 -l num_proc=1,mem_free=1G,h_rt=24:00:00" \
    "--init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --num_topics=5"
bash run_gibbs_qsub_wrapper.sh ../../../data/txt/twitter null 31-t10 \
    "-q text.q -t 1-5 -tc 2 -l num_proc=1,mem_free=1G,h_rt=24:00:00" \
    "--init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --num_topics=10"
bash run_gibbs_qsub_wrapper.sh ../../../data/txt/twitter null 31-t20 \
    "-q text.q -t 1-5 -tc 2 -l num_proc=1,mem_free=1G,h_rt=24:00:00" \
    "--init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --num_topics=20"
bash run_gibbs_qsub_wrapper.sh ../../../data/txt/twitter null 31-t40 \
    "-q text.q -t 1-5 -tc 2 -l num_proc=1,mem_free=2G,h_rt=24:00:00" \
    "--init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --num_topics=40"
bash run_gibbs_qsub_wrapper.sh ../../../data/txt/twitter null 31-t60 \
    "-q text.q -t 1-5 -tc 2 -l num_proc=1,mem_free=3G,h_rt=24:00:00" \
    "--init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --num_topics=60"
