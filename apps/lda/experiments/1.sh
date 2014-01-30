#!/bin/bash
for dataset in diff3 sim3 rel3
do
    if [ $dataset == diff3 ]
    then
        init_num_docs=167
        init_num_iters=200
    elif [ $dataset == sim3 ]
    then
        init_num_docs=177
        init_num_iters=200
    elif [ $dataset == rel3 ]
    then
        init_num_docs=158
        init_num_iters=200
    fi
    qsub -q text.q -tc 2 -l num_proc=1,mem_free=200M,h_rt=2:00:00 \
        run_lda.qsub $dataset --reservoir_size=0 \
        --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters
    qsub -q text.q -tc 2 -l num_proc=1,mem_free=200M,h_rt=2:00:00 \
        run_lda.qsub $dataset --reservoir_size=100 \
        --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters
    qsub -q text.q -tc 2 -l num_proc=1,mem_free=200M,h_rt=2:00:00 \
        run_lda.qsub $dataset --reservoir_size=1000 \
        --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters
    qsub -q text.q -tc 2 -l num_proc=1,mem_free=400M,h_rt=4:00:00 \
        run_lda.qsub $dataset --reservoir_size=10000 \
        --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters
    qsub -q text.q -tc 2 -l num_proc=1,mem_free=2G,h_rt=12:00:00 \
        run_lda.qsub $dataset --reservoir_size=100000 \
        --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters
    qsub -q text.q -tc 2 -l num_proc=1,mem_free=4G,h_rt=12:00:00 \
        run_lda.qsub $dataset --reservoir_size=500000 \
        --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters
done
