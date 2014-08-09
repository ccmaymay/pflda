#!/bin/bash

init_num_docs=0
init_num_iters=0

bash run_pf_qsub_wrapper.sh ../../../data/txt/gigaword null 34-p10-t50-rs1k-rss100-a0.1-b0.1 \
    "-q text.q -t 1-5 -tc 2 -l num_proc=1,mem_free=6G,h_rt=72:00:00" \
    "--init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --num_topics=50 --reservoir_size=1000 --rejuv_sample_size=100 --num_particles=10 --ess_threshold=4 --alpha=0.1 --beta=0.1"
bash run_pf_qsub_wrapper.sh ../../../data/txt/gigaword null 34-p1-t50-rs10k-rss1k-a0.1-b0.1 \
    "-q text.q -t 1-5 -tc 2 -l num_proc=1,mem_free=4G,h_rt=72:00:00" \
    "--init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --num_topics=50 --reservoir_size=10000 --rejuv_sample_size=1000 --num_particles=1 --ess_threshold=0.4 --alpha=0.1 --beta=0.1"
bash run_pf_qsub_wrapper.sh ../../../data/txt/gigaword null 34-p1-t50-rs1k-rss10-a0.1-b0.1 \
    "-q text.q -t 1-5 -tc 2 -l num_proc=1,mem_free=4G,h_rt=72:00:00" \
    "--init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --num_topics=50 --reservoir_size=1000 --rejuv_sample_size=10 --num_particles=1 --ess_threshold=0.4 --alpha=0.1 --beta=0.1"
bash run_pf_qsub_wrapper.sh ../../../data/txt/gigaword null 34-p1-t50-rs1k-rss10-a0.1-b0.01 \
    "-q text.q -t 1-5 -tc 2 -l num_proc=1,mem_free=4G,h_rt=72:00:00" \
    "--init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --num_topics=50 --reservoir_size=1000 --rejuv_sample_size=10 --num_particles=1 --ess_threshold=0.4 --alpha=0.1 --beta=0.01"
bash run_pf_qsub_wrapper.sh ../../../data/txt/gigaword null 34-p1-t50-rs1k-rss10-a0.01-b0.1 \
    "-q text.q -t 1-5 -tc 2 -l num_proc=1,mem_free=4G,h_rt=72:00:00" \
    "--init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --num_topics=50 --reservoir_size=1000 --rejuv_sample_size=10 --num_particles=1 --ess_threshold=0.4 --alpha=0.01 --beta=0.1"
bash run_pf_qsub_wrapper.sh ../../../data/txt/gigaword null 34-p1-t50-rs1k-rss10-a0.01-b0.01 \
    "-q text.q -t 1-5 -tc 2 -l num_proc=1,mem_free=4G,h_rt=72:00:00" \
    "--init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --num_topics=50 --reservoir_size=1000 --rejuv_sample_size=10 --num_particles=1 --ess_threshold=0.4 --alpha=0.01 --beta=0.01"
bash run_pf_qsub_wrapper.sh ../../../data/txt/gigaword null 34-p1-t100-rs1k-rss100-a0.1-b0.1 \
    "-q text.q -t 1-5 -tc 2 -l num_proc=1,mem_free=4G,h_rt=72:00:00" \
    "--init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --num_topics=100 --reservoir_size=1000 --rejuv_sample_size=100 --num_particles=1 --ess_threshold=0.4 --alpha=0.1 --beta=0.1"
bash run_pf_qsub_wrapper.sh ../../../data/txt/gigaword null 34-p1-t50-rs1k-rss100-a0.1-b0.1 \
    "-q text.q -t 1-5 -tc 2 -l num_proc=1,mem_free=4G,h_rt=72:00:00" \
    "--init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --num_topics=50 --reservoir_size=1000 --rejuv_sample_size=100 --num_particles=1 --ess_threshold=0.4 --alpha=0.1 --beta=0.1"
bash run_pf_qsub_wrapper.sh ../../../data/txt/gigaword null 34-p1-t10-rs1k-rss100-a0.1-b0.1 \
    "-q text.q -t 1-5 -tc 2 -l num_proc=1,mem_free=4G,h_rt=72:00:00" \
    "--init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --num_topics=10 --reservoir_size=1000 --rejuv_sample_size=100 --num_particles=1 --ess_threshold=0.4 --alpha=0.1 --beta=0.1"
