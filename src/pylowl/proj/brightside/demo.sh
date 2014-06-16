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
    --init_samples=10 \
    --test_samples=10 \
    --max_time=30 \
    --output_dir="$OUTPUT_DIR"

first_graph=true
for topics_f in `ls -t "$OUTPUT_DIR"/*.topics`
do
    if [ -f "$topics_f" ] # guard against '*.topics' in empty case
    then
        python -m pylowl.proj.brightside.postproc.generate_d3_topic_graph \
            "$TRUNC" "$DATA_DIR/vocab" "${topics_f}" "${topics_f}.json"
        if $first_graph
        then
            cp "${topics_f}.json" "$OUTPUT_DIR/graph.json"
        fi
        first_graph=false
    fi
done

python -m pylowl.proj.brightside.postproc.generate_d3_subgraphs \
    "$OUTPUT_DIR/log" "$OUTPUT_DIR/subgraphs.json"

ln -s "$SRC_DIR/postproc/d3.v3.js" "$SRC_DIR/postproc/graph.html" "$SRC_DIR/postproc/subgraphs.html" "$OUTPUT_DIR/"

cd "$OUTPUT_DIR"
python -m SimpleHTTPServer 8000
