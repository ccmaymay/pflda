#!/bin/bash
mkdir -p 1-rs0
qsub -q text.q -l num_proc=1,mem_free=200M,h_rt=1:00:00 run_tng3.qsub 1-rs0 --reservoir_size=0
mkdir -p 1-rs100
qsub -q text.q -l num_proc=1,mem_free=200M,h_rt=1:00:00 run_tng3.qsub 1-rs100 --reservoir_size=100
mkdir -p 1-rs1k
qsub -q text.q -l num_proc=1,mem_free=200M,h_rt=1:00:00 run_tng3.qsub 1-rs1k --reservoir_size=1000
mkdir -p 1-rs10k
qsub -q text.q -l num_proc=1,mem_free=400M,h_rt=2:00:00 run_tng3.qsub 1-rs10k --reservoir_size=10000
mkdir -p 1-rs100k
qsub -q text.q -l num_proc=1,mem_free=2G,h_rt=6:00:00 run_tng3.qsub 1-rs100k --reservoir_size=100000
mkdir -p 1-rs500k
qsub -q text.q -l num_proc=1,mem_free=4G,h_rt=6:00:00 run_tng3.qsub 1-rs500k --reservoir_size=500000
