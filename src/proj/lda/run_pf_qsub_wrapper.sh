#!/bin/bash

dataset_path="$1"
dataset_name="$2"
experiment="$3"
d="$experiment/$dataset_name"

mkdir -p "$d"
qsub $4 -o "$d/\$TASK_ID.log" run_pf.qsub $dataset_path $dataset_name $5
