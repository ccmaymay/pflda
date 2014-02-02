#!/bin/bash

experiment="$1"
for dataset in diff3 rel3 sim3
do
    bash qsub.sh $dataset $experiment "$2" "$3"
done
