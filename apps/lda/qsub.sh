#!/bin/bash

script_path="$1"
dataset_path="$2"
dataset_name="$3"
experiment="$4"
d="$experiment/$dataset_name"

mkdir -p "$d"
for i in {1..120}
do
    f="$d/${i}.log"
    printenv > "$f"
    bash "$script_path" $dataset_path $dataset_name $6 >> "$f" 2>&1
done
