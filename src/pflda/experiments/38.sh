#!/bin/bash

# 20 with single particle, new dataset path

dataset_path=../../../../data/txt/tng-noheader-nowalls-nonalpha
for dataset_name in diff3 sim3 rel3
do
    if [ $dataset_name == diff3 ]
    then
        init_num_docs=167
        init_num_iters=200
        init_seed=209049874 # 0.584990
    elif [ $dataset_name == sim3 ]
    then
        init_num_docs=177
        init_num_iters=200
        init_seed=225706106 # 0.226941
    elif [ $dataset_name == rel3 ]
    then
        init_num_docs=158
        init_num_iters=200
        init_seed=883999000 # 0.129669
    fi

    bash run_pf_qsub_wrapper.sh $dataset_path $dataset_name 20-rs0 \
        "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
        "--num_particles=1 --ess_threshold=2 --reservoir_size=0 --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --init_seed=$init_seed"
    bash run_pf_qsub_wrapper.sh $dataset_path $dataset_name 20-rs10 \
        "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
        "--num_particles=1 --ess_threshold=2 --reservoir_size=10 --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --init_seed=$init_seed"
    bash run_pf_qsub_wrapper.sh $dataset_path $dataset_name 20-rs100 \
        "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
        "--num_particles=1 --ess_threshold=2 --reservoir_size=100 --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --init_seed=$init_seed"
    bash run_pf_qsub_wrapper.sh $dataset_path $dataset_name 20-rs1k \
        "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=200M,h_rt=1:00:00" \
        "--num_particles=1 --ess_threshold=2 --reservoir_size=1000 --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --init_seed=$init_seed"
    bash run_pf_qsub_wrapper.sh $dataset_path $dataset_name 20-rs10k \
        "-q text.q -t 1-30 -tc 2 -l num_proc=1,mem_free=500M,h_rt=4:00:00" \
        "--num_particles=1 --ess_threshold=2 --reservoir_size=10000 --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --init_seed=$init_seed"
    bash run_pf_qsub_wrapper.sh $dataset_path $dataset_name 20-rs100k \
        "-q text.q -t 1-30 -tc 4 -l num_proc=1,mem_free=2G,h_rt=24:00:00" \
        "--num_particles=1 --ess_threshold=2 --reservoir_size=100000 --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --init_seed=$init_seed"
    bash run_pf_qsub_wrapper.sh $dataset_path $dataset_name 20-rs500k \
        "-q text.q -t 1-30 -tc 8 -l num_proc=1,mem_free=4G,h_rt=36:00:00" \
        "--num_particles=1 --ess_threshold=2 --reservoir_size=500000 --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters --init_seed=$init_seed"
done
