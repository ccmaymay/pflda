#!/bin/bash

if [ $# -ne 1 ]
then
    echo 'Specify brightside output base dir.' >&2
    exit 1
fi

OUTPUT_BASE_DIR="$1"

for d in "$OUTPUT_BASE_DIR"/*
do
    echo "$d"
done
