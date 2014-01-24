#!/bin/bash

log_stem="$1"
shift
n="$1"
shift

i=0
while [ "$i" -lt "$n" ]
do
    bash run_lda.sh $@ > "$log_stem.$i.log"
done
