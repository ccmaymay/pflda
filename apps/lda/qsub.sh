#!/bin/bash

dataset="$1"
experiment="$2"
d="$experiment/$dataset"

mkdir -p "$d"
qsub -o "$d/\$TASK_ID.log" $3 run_lda.qsub $dataset $4
