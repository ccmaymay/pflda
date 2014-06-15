#!/bin/bash

set -e

DATA_DIR="../data/tng.rasp"
VOCAB_FILENAME="$DATA_DIR/vocab"
TRAIN_FILENAME="$DATA_DIR/train"
TEST_FILENAME="$DATA_DIR/test"
VOCAB_SIZE=`cat "$VOCAB_FILENAME" | wc -l`
NUM_DOCS=`cat "$TRAIN_FILENAME" | wc -l`
PYTHON_SCRIPT=run_m0.py
OUTPUT_BASE_DIR="../output"

mkdir -p "$OUTPUT_BASE_DIR"
OUTPUT_DIR=`mktemp -d "$OUTPUT_BASE_DIR/XXXXXX"`

TRUNC="$1"
shift

# TODO max iter

python "$PYTHON_SCRIPT" \
    --directory="$OUTPUT_DIR" \
    --data_path="$TRAIN_FILENAME" \
    --test_data_path="$TEST_FILENAME" \
    --max_iter=100 \
    --D="$NUM_DOCS" \
    --W="$VOCAB_SIZE" \
    --trunc="$TRUNC" \
    "$@"

bash generate_topic_graphs.sh "$TRUNC" "$VOCAB_FILENAME" "$OUTPUT_DIR"
python -m output.generate_d3_subgraphs "$OUTPUT_DIR/log" "$OUTPUT_DIR/subgraphs.json"
