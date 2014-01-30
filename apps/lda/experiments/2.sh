#!/bin/bash

bash qsub_tng3.sh 2-rs1k-ibs0 -q text.q -tc 2 \
    -l num_proc=1,mem_free=200M,h_rt=2:00:00 \
    run_lda.qsub $dataset --reservoir_size=1000 \
    --init_num_docs=0 --init_num_iters=0
bash qsub_tng3.sh 2-rs1k-ibs10 -q text.q -tc 2 \
    -l num_proc=1,mem_free=200M,h_rt=2:00:00 \
    run_lda.qsub $dataset --reservoir_size=1000 \
    --init_num_docs=10 --init_num_iters=10
bash qsub_tng3.sh 2-rs1k-ibs30 -q text.q -tc 2 \
    -l num_proc=1,mem_free=200M,h_rt=2:00:00 \
    run_lda.qsub $dataset --reservoir_size=1000 \
    --init_num_docs=30 --init_num_iters=30
bash qsub_tng3.sh 2-rs1k-ibs100 -q text.q -tc 2 \
    -l num_proc=1,mem_free=200M,h_rt=2:00:00 \
    run_lda.qsub $dataset --reservoir_size=1000 \
    --init_num_docs=100 --init_num_iters=100
bash qsub_tng3.sh 2-rs1k-ibs300 -q text.q -tc 2 \
    -l num_proc=1,mem_free=200M,h_rt=2:00:00 \
    run_lda.qsub $dataset --reservoir_size=1000 \
    --init_num_docs=300 --init_num_iters=300
bash qsub_tng3.sh 2-rs1k-ibs1k -q text.q -tc 2 \
    -l num_proc=1,mem_free=200M,h_rt=2:00:00 \
    run_lda.qsub $dataset --reservoir_size=1000 \
    --init_num_docs=1000 --init_num_iters=1000
