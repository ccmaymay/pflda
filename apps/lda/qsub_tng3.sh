#!/bin/bash

experiment_dir="$1"
shift

for dataset in diff3 rel3 sim3
do
    qsub -o "$experiment_dir/dataset/\$TASK_ID.log" "$@"
done
