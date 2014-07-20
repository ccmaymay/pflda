#!/bin/bash
#$ -cwd
#$ -j y
#$ -V
#$ -N "import_tng"
#$ -q text.q
#$ -l num_proc=1,mem_free=1G,h_rt=12:00:00

for t in train test
do
    python -m pylowl.proj.brightside.preproc.tng_to_concrete ~/20news-bydate-$t data/txt/tng.concrete/$t --remove_non_ascii --remove_emails --remove_walls --remove_first_paragraph --remove_email_headers --remove_email_history --remove_writes_lines
done
