#!/bin/bash

output_dir="$1"
shift

for dataset in diff3 sim3 rel3
do
    bash run_n_bg.sh "$output_dir/$dataset" 30 $dataset $@
done
