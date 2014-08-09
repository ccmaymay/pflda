#!/bin/bash
for d in {1..99} {1..99}-*
do
    if [ -d $d ]
    then
        echo $d
        if [ "$d" == 22 -o "$d" == 23 -o "$d" == 24 -o "$d" == 26 ]
        then
            # gibbs
            python extract_log_stats.py $d iter
        else
            # pf
            python extract_log_stats.py $d doc
        fi
    fi
done
