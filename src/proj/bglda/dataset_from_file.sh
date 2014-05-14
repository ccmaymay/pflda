#!/bin/bash

TRAIN_FRAC=0.6

if [ $# -ne 2 ]
then
    echo 'Specify input filename and output dataset path.' >&2
    exit 1
fi

filename="$1"
dataset_path="$2"

temp_data=`mktemp dataset_from_file.XXXXXX`

mkdir -p "$dataset_path/train" "$dataset_path/test"

sed 's/^/X Y /' "$filename" | shuf > "$temp_data"
num_docs=`cat "$temp_data" | wc -l`
python_calc='print int('"$TRAIN_FRAC"' * '"$num_docs"')'
num_train_docs=`python -c "$python_calc"`
num_test_docs=$(($num_docs - $num_train_docs))

raw_train_path="$dataset_path/train/all"
head -n "$num_train_docs" "$temp_data" > "$raw_train_path"
gzip "$raw_train_path"
raw_test_path="$dataset_path/test/all"
tail -n "$num_test_docs" "$temp_data" > "$raw_test_path"
gzip "$raw_test_path"

rm -f "$temp_data"
