#!/bin/bash

dataset_path="$1"
experiment="$2"

for dataset_name in diff3 rel3 sim3
do
    bash run_pf_qsub_wrapper.sh $dataset_path $dataset_name $experiment "$3" "$4"
done
