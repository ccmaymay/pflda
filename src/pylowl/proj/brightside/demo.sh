#!/bin/bash

set -e

cd ../../../../ # repo root

export PYTHONPATH=build/lib.linux-x86_64-2.7

SRC_DIR=src/pylowl/proj/brightside
DATA_DIR=data/txt/tng.rasp
OUTPUT_BASE_DIR=output/pylowl/proj/brightside
TRUNC=1,3,3

mkdir -p "$OUTPUT_BASE_DIR"
OUTPUT_DIR=`mktemp -d "$OUTPUT_BASE_DIR/XXXXXX"`

python setup.py build --with-proj-brightside

python -m pylowl.proj.brightside.run_m0 \
    --trunc="$TRUNC" \
    --data_path="$DATA_DIR/train" \
    --test_data_path="$DATA_DIR/test" \
    --save_model \
    --init_samples=50 \
    --test_samples=50 \
    --max_time=60 \
    --batchsize=20 \
    --output_dir="$OUTPUT_DIR"

topics_f=`ls -t "$OUTPUT_DIR"/*.topics | head -n 1`
if [ -f "$topics_f" ]
then
    python -m pylowl.proj.brightside.postproc.generate_d3_topic_graph \
        "$TRUNC" "$DATA_DIR/vocab" "${topics_f}" "$OUTPUT_DIR/graph.json"
fi

python -m pylowl.proj.brightside.postproc.generate_d3_subgraphs \
    "$OUTPUT_DIR/log" "$OUTPUT_DIR/subgraphs.json"

ln -s "$PWD/$SRC_DIR/postproc/d3.v3.js" "$PWD/$SRC_DIR/postproc/graph.html" "$PWD/$SRC_DIR/postproc/subgraphs.html" "$OUTPUT_DIR/"

cd "$OUTPUT_DIR"
python -m SimpleHTTPServer 8000
