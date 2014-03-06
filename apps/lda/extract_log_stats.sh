#!/bin/bash
for d in {1..100} {1..100}-*
do
    if [ -d $d ]
    then
        echo $d
        python extract_log_stats.py $d
    fi
done
