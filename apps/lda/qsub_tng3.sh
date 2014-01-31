#!/bin/bash

for dataset in diff3 rel3 sim3
do
    d="$1/$dataset"
    mkdir -p "$d"
    qsub -o "$d/\$TASK_ID.log" $2 run_lda.qsub $dataset $3
done
