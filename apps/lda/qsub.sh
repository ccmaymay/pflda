#!/bin/bash

dataset_path="$1"
dataset_name="$2"
experiment="$3"
d="$experiment/$dataset_name"

mkdir -p "$d"
for i in {1..120}
do
    f="$d/${i}.log"
    printenv > "$f"
    bash run_gibbs.sh $dataset_path $dataset_name $5 >> "$f" 2>&1
done
