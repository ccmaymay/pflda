#!/bin/bash
for dataset in diff3 sim3 rel3
do
    bash file_dataset_logs.sh $dataset "$@"
done
