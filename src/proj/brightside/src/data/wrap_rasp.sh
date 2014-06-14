#!/bin/bash

if [ $# -ne 3 ]
then
    echo 'specify path to rasp.sh, input file path, and output file path' >&2
    exit 1
fi

SCRIPT_DIR=`dirname "$0"`
RASP_SH="$1"
INPUT_PATH="$2"
OUTPUT_PATH="$3"

rasp_parse=cat "$RASP_SH" < "$INPUT_PATH" > "$OUTPUT_PATH"
