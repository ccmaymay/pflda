#!/bin/bash
for d in {1..100} {1..100}-*
do
    if [ -d $d ]
    then
        echo $d
        python $d
    fi
done
