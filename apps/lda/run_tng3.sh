#!/bin/bash

output_dir="$1"
shift

for dataset in diff3 sim3 rel3
do
    log_file=`mktemp "$output_dir/$dataset/XXXXXX.log"`
    bash run_lda.sh $dataset $@ > "$log_file" 2>&1
done
