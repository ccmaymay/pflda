#!/bin/bash

dataset_path="$1"
dataset_name="$2"
experiment="$3"
d="$experiment/$dataset"

mkdir -p "$d"
qsub -o "$d/\$TASK_ID.log" $4 run_lda.qsub $dataset_path $dataset_name $5
