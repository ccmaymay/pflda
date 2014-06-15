#!/bin/bash

if [ $# -ne 3 ]
then
    echo 'Specify trunc csv, vocab filename, and output dir.' >&2
    exit 1
fi

trunc_csv="$1"
vocab_filename="$2"
output_dir="$3"

topic_types_fn=`mktemp generate_topic_graphs.top.XXXXXX`

for f in "$output_dir"/*.topics
do
    python -m output.write_topic_types "$vocab_filename" "$f" "$topic_types_fn"
    python -m output.generate_d3_topic_graph "$trunc_csv" "$topic_types_fn" "${f}.json"
done

rm -f "$topic_types_fn"
