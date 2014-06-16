#!/bin/bash

set -e

cd ../../../../ # repo root

export PYTHONPATH=build/lib.linux-x86_64-2.7

python setup.py build --with-proj-brightside

SRC_DIR=src/pylowl/proj/brightside

DATA_DIR=`mktemp -d data/txt/demo.XXXXXX`
VOCAB_PATH=`mktemp data/txt/demo.vocab.XXXXXX`
echo "Writing data to $DATA_DIR ..."
bash "$SRC_DIR/make_demo_data.sh" src "$DATA_DIR"
echo "Writing vocab to $VOCAB_PATH ..."
python -m pylowl.proj.brightside.preproc.extract_concrete_vocab \
    "$DATA_DIR"/'*' "$VOCAB_PATH"

OUTPUT_BASE_DIR=output/pylowl/proj/brightside
echo "Creating output dir within $OUTPUT_BASE_DIR ..."
mkdir -p "$OUTPUT_BASE_DIR"
OUTPUT_DIR=`mktemp -d "$OUTPUT_BASE_DIR/XXXXXX"`

WEB_PORT=8000

TRUNC=1,2,2

python -m pylowl.proj.brightside.run_m0 \
    --trunc="$TRUNC" \
    --concrete \
    --concrete_vocab_path="$VOCAB_PATH" \
    --data_path="$DATA_DIR"/'*' \
    --test_data_path="$DATA_DIR"/'*' \
    --save_model \
    --init_samples=50 \
    --test_samples=50 \
    --max_time=60 \
    --batchsize=20 \
    --output_dir="$OUTPUT_DIR"

echo "Extracting graph data in $OUTPUT_DIR ..."
topics_f=`ls -t "$OUTPUT_DIR"/*.topics | head -n 1`
if [ -f "$topics_f" ]
then
    python -m pylowl.proj.brightside.postproc.generate_d3_topic_graph \
        "$TRUNC" "$VOCAB_PATH" "${topics_f}" "$OUTPUT_DIR/graph.json"
fi

python -m pylowl.proj.brightside.postproc.generate_d3_subgraphs \
    "$OUTPUT_DIR/log" "$OUTPUT_DIR/subgraphs.json"

echo "Linking graph visualization code to $OUTPUT_DIR ..."
ln -s "$PWD/$SRC_DIR/postproc/d3.v3.js" "$PWD/$SRC_DIR/postproc/graph.html" "$PWD/$SRC_DIR/postproc/subgraphs.html" "$OUTPUT_DIR/"

echo "Launching web server in $OUTPUT_DIR on port $WEB_PORT ..."
cd "$OUTPUT_DIR"
python -m SimpleHTTPServer $WEB_PORT
