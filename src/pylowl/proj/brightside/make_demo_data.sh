#!/bin/bash

set -e

if [ $# -ne 2 ]
then
    echo 'specify input and output dir' >&2
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

temp_output_dir=`mktemp -d demo.temp.XXXXXX`
mkdir -p "$temp_output_dir"
find "$INPUT_DIR" -type f -name '*.py' \
    -or -name '*.sh' \
    -or -name '*.c' \
    -or -name '*.h' \
    -or -name '*.pxd' \
    -or -name '*.pyx' \
    | xargs -n 1 cp --backup=numbered -t "$temp_output_dir"
python -m pylowl.proj.brightside.preproc.docs_to_concrete "$temp_output_dir"/'*' "$OUTPUT_DIR"
rm -rf "$temp_output_dir"
