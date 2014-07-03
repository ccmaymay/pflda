#!/bin/bash
#$ -cwd
#$ -j y
#$ -V
#$ -N "postproc"
#$ -q text.q
#$ -l num_proc=1,mem_free=2G,h_rt=1:00:00

if [ $# -ne 3 ]
then
    echo 'Specify output dir, truncation csv, and vocab path.' >&2
    exit 1
fi

OUTPUT_DIR="$1"
TRUNC="$2"
VOCAB_PATH="$3"

python -m pylowl.proj.brightside.postproc.generate_d3_topic_graph \
    --lambda_ss "$OUTPUT_DIR/final.lambda_ss" \
    --Elogpi "$OUTPUT_DIR/final.Elogpi" \
    --logEpi "$OUTPUT_DIR/final.logEpi" \
    --Elogtheta "$OUTPUT_DIR/final.Elogtheta" \
    --logEtheta "$OUTPUT_DIR/final.logEtheta" \
    "$TRUNC" "$VOCAB_PATH" "$OUTPUT_DIR/graph.json"

python -m pylowl.proj.brightside.postproc.generate_d3_subgraphs \
    --subtree "$OUTPUT_DIR/subtree" \
    --lambda_ss "$OUTPUT_DIR/subtree_lambda_ss" \
    --Elogpi "$OUTPUT_DIR/subtree_Elogpi" \
    --logEpi "$OUTPUT_DIR/subtree_logEpi" \
    --Elogtheta "$OUTPUT_DIR/subtree_Elogtheta" \
    --logEtheta "$OUTPUT_DIR/subtree_logEtheta" \
    "$TRUNC" "$VOCAB_PATH" "$OUTPUT_DIR/subgraphs.json"
