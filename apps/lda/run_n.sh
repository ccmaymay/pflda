#!/bin/bash

log_dir="$1"
shift
n="$1"
shift

i=0
mkdir -p "$log_dir"
while [ "$i" -lt "$n" ]
do
    bash run_lda.sh $@ > "$log_dir/$i.log"
    i=$(($i + 1))
done
