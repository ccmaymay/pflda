#!/bin/bash

set -e

cd ../../../.. # repository root

OUTPUT_DIR=`mktemp -d output/pylowl/proj/brightside/XXXXXX`
TRUNC=1,5,4,3
VOCAB_PATH=data/txt/tng.rasp.concrete.catuser/vocab

rm -rf "$OUTPUT_DIR"/*

python setup.py build --with-proj-brightside

python -m pylowl.proj.brightside.run \
    --trunc="$TRUNC" \
    --data_path='data/txt/tng.rasp.concrete.catuser/train/*' \
    --test_data_path='data/txt/tng.rasp.concrete.catuser/test/*' \
    --test_samples=100 \
    --init_samples=200 \
    --max_time=600 \
    --save_model \
    --output_dir="$OUTPUT_DIR" \
    --concrete \
    --concrete_vocab_path="$VOCAB_PATH" \
    --U=20 \
    --D=11222 \
    --W=4571 \
    --user_doc_reservoir_capacity=100 \
    --user_subtree_selection_interval=50 \
    --log_level=DEBUG

bash src/pylowl/proj/brightside/postproc/generate_d3_inputs.sh \
    "$OUTPUT_DIR" \
    "$TRUNC" \
    "$VOCAB_PATH"

bash src/pylowl/proj/brightside/postproc/symlink_viz_resources.sh \
    "$OUTPUT_DIR"

echo "$OUTPUT_DIR"
