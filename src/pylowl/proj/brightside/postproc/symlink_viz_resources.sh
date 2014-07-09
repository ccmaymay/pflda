#!/bin/bash

if [ $# -ne 1 ]
then
    echo 'Specify output directory.' >&2
    exit 1
fi

OUTPUT_DIR="$1"
POSTPROC_DIR=`python -c "import os; print os.path.abspath(os.path.dirname(\"$0\"))"`

for f in graph.html subgraphs.html d3.v3.js
do
    ln -s "$POSTPROC_DIR/$f" "$OUTPUT_DIR/"
done
