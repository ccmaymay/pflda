#!/bin/bash

set -e

# cd to repository root
cd ../../../../

# put build dir in python path
DISTUTILS_PLATFORM=`python -c 'import distutils.util; print distutils.util.get_platform()'`
DISTUTILS_VERSION=`python -c 'import sys; print "%s.%s" % sys.version_info[:2]'`
export PYTHONPATH="build/lib.${DISTUTILS_PLATFORM}-${DISTUTILS_VERSION}:$PYTHONPATH"

# build pylowl with brightside
python setup.py build --with-proj-brightside

SRC_DIR=src/pylowl/proj/brightside

# make toy dataset from source code
DATA_DIR=`mktemp -d data/txt/demo.XXXXXX`
VOCAB_PATH=`mktemp data/txt/demo.vocab.XXXXXX`
echo "Writing data to $DATA_DIR ..."
bash "$SRC_DIR/make_demo_data.sh" src "$DATA_DIR"
echo "Writing vocab to $VOCAB_PATH ..."
python -m pylowl.proj.brightside.preproc.extract_concrete_vocab \
    "$DATA_DIR"/'*' 0 0 0 "$VOCAB_PATH"

# create random output directory
OUTPUT_BASE_DIR=output/pylowl/proj/brightside
echo "Creating output dir within $OUTPUT_BASE_DIR ..."
mkdir -p "$OUTPUT_BASE_DIR"
OUTPUT_DIR=`mktemp -d "$OUTPUT_BASE_DIR/XXXXXX"`

# truncated tree per-level per-node widths
# (e.g., 1 root, 2 children below root, 2 children below each of them)
TRUNC=1,2,2

# run online variational inference for nhdp:
# set PYTHONOPTIMIZE=1 to prevent assertions from firing (and failing
# on grid)
# use all data in $DATA_DIR for both training and testing
# use 50 documents in initialization
# use 50 documents for testing
# use 20-document mini-batches
# run for at most 60 seconds (plus overhead)
PYTHONOPTIMIZE=1 python -m pylowl.proj.brightside.run_m0 \
    --trunc="$TRUNC" \
    --concrete_vocab_path="$VOCAB_PATH" \
    --data_path="$DATA_DIR"/'*' \
    --test_data_path="$DATA_DIR"/'*' \
    --save_model \
    --init_samples=50 \
    --test_samples=50 \
    --max_time=60 \
    --batchsize=20 \
    --output_dir="$OUTPUT_DIR"

# extract global and local graph data (for visualization) from output
echo "Extracting graph data in $OUTPUT_DIR ..."
bash "$SRC_DIR/postproc/generate_d3_inputs.sh" "$OUTPUT_DIR" "$TRUNC" "$VOCAB_PATH"

# put d3 and html files in $OUTPUT_DIR (so we don't have to change the
# relative locations of the json files they load)
echo "Linking graph visualization code to $OUTPUT_DIR ..."
ln -s "$PWD/$SRC_DIR/postproc/d3.v3.js" "$PWD/$SRC_DIR/postproc/graph.html" "$PWD/$SRC_DIR/postproc/subgraphs.html" "$OUTPUT_DIR/"

echo

WEB_PORT=8000

echo "Copy $PWD/$OUTPUT_DIR"
echo "somewhere a web server can see it and point a browser to it,"
echo "or do:"
echo "cd \"$PWD/$OUTPUT_DIR\" && python -m SimpleHTTPServer $WEB_PORT"

# start a simple web server in $OUTPUT_DIR
#echo "Launching web server in $OUTPUT_DIR on port $WEB_PORT ..."
#cd "$OUTPUT_DIR"
#python -m SimpleHTTPServer $WEB_PORT
