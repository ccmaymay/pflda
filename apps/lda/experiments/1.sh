#!/bin/bash
for dataset in diff3 sim3 rel3
do
    qsub -q text.q -l num_proc=1,mem_free=200M,h_rt=1:00:00 run_lda.qsub $dataset --reservoir_size=0
    qsub -q text.q -l num_proc=1,mem_free=200M,h_rt=1:00:00 run_lda.qsub $dataset --reservoir_size=100
    qsub -q text.q -l num_proc=1,mem_free=200M,h_rt=1:00:00 run_lda.qsub $dataset --reservoir_size=1000
    qsub -q text.q -l num_proc=1,mem_free=400M,h_rt=2:00:00 run_lda.qsub $dataset --reservoir_size=10000
    qsub -q text.q -l num_proc=1,mem_free=2G,h_rt=6:00:00 run_lda.qsub $dataset --reservoir_size=100000
    qsub -q text.q -l num_proc=1,mem_free=4G,h_rt=6:00:00 run_lda.qsub $dataset --reservoir_size=500000
done
