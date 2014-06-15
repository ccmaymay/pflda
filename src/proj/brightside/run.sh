#!/bin/bash

set -e

DATA_DIR="../../../data/txt/tng.rasp"
VOCAB_FILENAME="$DATA_DIR/vocab"
TRAIN_FILENAME="$DATA_DIR/train"
TEST_FILENAME="$DATA_DIR/test"
PYTHON_SCRIPT=run_m0.py
OUTPUT_BASE_DIR="../../../output/proj/brightside"

mkdir -p "$OUTPUT_BASE_DIR"
OUTPUT_DIR=`mktemp -d "$OUTPUT_BASE_DIR/XXXXXX"`

TRUNC="$1"
shift

python "$PYTHON_SCRIPT" \
    --directory="$OUTPUT_DIR" \
    --data_path="$TRAIN_FILENAME" \
    --test_data_path="$TEST_FILENAME" \
    --trunc="$TRUNC" \
    "$@"

first_graph=true
for topics_f in `ls -t "$OUTPUT_DIR"/*.topics`
do
    if [ -f "$topics_f" ] # guard against '*.topics' in empty case
    then
        python -m output.generate_d3_topic_graph "$TRUNC" "$VOCAB_FILENAME" "${topics_f}" "${topics_f}.json"
        if $first_graph
        then
            cp "${topics_f}.json" "$OUTPUT_DIR/graph.json"
        fi
        first_graph=false
    fi
done
python -m output.generate_d3_subgraphs "$OUTPUT_DIR/log" "$OUTPUT_DIR/subgraphs.json"

ln -s "$PWD/output/d3.v3.js" "$PWD/output/graph.html" "$PWD/output/subgraphs.html" "$OUTPUT_DIR/"

#cd "$OUTPUT_DIR"
#python -m SimpleHTTPServer 8000
