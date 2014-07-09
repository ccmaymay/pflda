#!/bin/bash

set -e

cd ../../../.. # repository root

TRUNC=1,5,4
VOCAB_PATH=data/txt/tng.rasp.concrete.catuser/vocab

echo "Creating output directory..."
OUTPUT_DIR=`mktemp -d output/pylowl/proj/brightside/XXXXXX`

echo "Building brightside..."
python setup.py build --with-proj-brightside

echo "Running stochastic variational inference..."
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

echo "Generating D3 inputs..."
bash src/pylowl/proj/brightside/postproc/generate_d3_inputs.sh \
    "$OUTPUT_DIR" \
    "$TRUNC" \
    "$VOCAB_PATH"

echo "Linking visualization code to output directory..."
bash src/pylowl/proj/brightside/postproc/symlink_viz_resources.sh \
    "$OUTPUT_DIR"

echo "Done:"
echo "$OUTPUT_DIR"
