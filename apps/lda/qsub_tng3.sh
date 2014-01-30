#!/bin/bash

experiment_dir="$1"
shift

for dataset in diff3 rel3 sim3
do
    d="$experiment_dir/$dataset"
    mkdir -p "$d"
    qsub -o "$d/\$TASK_ID.log" "$@"
done
