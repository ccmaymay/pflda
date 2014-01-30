#!/bin/bash

dataset="$1"
shift
current_job_num=

for fn in slda.o*.*
do
    job_num=`echo "$fn" | cut -d. -f 2`
    if ! [ -z "$current_job_num" ] && [ "$current_job_num" != "$job_num" ]
    then
        shift
        if [ -z "$1" ]
        then
            exit 0
        fi
    fi
    current_job_num="$job_num"
    current_experiment_group="$1"

    echo "$current_experiment_group" "$current_job_num"

    d="$current_experiment_group/$dataset"
    mkdir -p "$d"
    mv "$fn" "$d/"
done
