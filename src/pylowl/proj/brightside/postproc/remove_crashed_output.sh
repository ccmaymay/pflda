#!/bin/bash

if [ $# -lt 1 ]
then
    echo 'Specify brightside output base dir.' >&2
    exit 1
fi

OUTPUT_BASE_DIR="$1"
BASENAMES="graph.json"

if [ "$2" == "-y" ]
then
    ACTUALLY_REMOVE=true
    echo "Removing crashed output directories..."
else
    ACTUALLY_REMOVE=false
    echo "Not removing crashed output directories."
    echo "To remove, specify -y as the second argument."
fi

for model_dir in "$OUTPUT_BASE_DIR"/*
do
    if [ -d "$model_dir" ]
    then
        echo
        echo "$model_dir"
        for run_dir in "$model_dir"/*
        do
            remove=false
            for bn in $BASENAMES
            do
                path="$run_dir/$bn"
                if ! [ -f "$path" ]
                then
                    remove=true
                fi
            done

            if $remove
            then
                echo "- $run_dir"
                if $ACTUALLY_REMOVE
                then
                    rm -rf "$run_dir"
                fi
            else
                echo "+ $run_dir"
            fi
        done
    fi
done
