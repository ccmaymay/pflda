#!/bin/bash

# find jerboa at https://github.com/vandurme/jerboa

jerboa_jar_path="$1"
shift
jerboa_unicode_csv_path="$1"
shift
stop_list="$1"
shift
input_filename="$1"
shift
output_filename="$1"

temp1=`mktemp tokenize_twitter.XXXXXX`
temp2=`mktemp tokenize_twitter.XXXXXX`

cut -f 6- < "$input_filename" > "$temp1"
java -DTwitterTokenizer.unicode="$jerboa_unicode_csv_path" -classpath "$jerboa_jar_path" edu.jhu.jerboa.processing.Tokenizer TWITTER "$temp1" > "$temp2"
python filter_twitter_tok.py --idf_threshold=0.0001 --stop_list="$stop_list" "$temp2" > "$output_filename"

rm -f "$temp1" "$temp2"
