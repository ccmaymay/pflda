#!/bin/bash

startup_interval=15
shutdown_interval=2

for i in `seq 100`
do
    for dataset_filename in tng-nonalpha tng-whitespace tng-whitespace-mc
    do
        for dataset_name in diff3 rel3 sim3
        do
            if [ $dataset_name == diff3 ]
            then
                init_num_docs=167
                init_num_iters=200
            elif [ $dataset_name == sim3 ]
            then
                init_num_docs=177
                init_num_iters=200
            elif [ $dataset_name == rel3 ]
            then
                init_num_docs=158
                init_num_iters=200
            fi

            log_filename=find_seed.${dataset_filename}.${dataset_name}.log
            python run_pf.py ../../../data/txt/$dataset_filename $dataset_name --init_num_docs=$init_num_docs --init_num_iters=$init_num_iters >> "$log_filename" &
            sleep $startup_interval
            pkill -f run_lda
            sleep $shutdown_interval
            echo >> "$log_filename"
            echo >> "$log_filename"
        done
    done
done
