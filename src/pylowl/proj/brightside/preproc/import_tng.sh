#!/bin/bash
#$ -cwd
#$ -j y
#$ -V
#$ -N "import_tng"
#$ -q text.q
#$ -l num_proc=1,mem_free=1G,h_rt=12:00:00

for t in train test
do
    python -m pylowl.proj.brightside.preproc.tng_to_concrete \
        $HOME/20news-bydate-$t \
        data/txt/tng.concrete.orig.split/$t \
        --remove_non_ascii \
        --remove_emails \
        --remove_walls \
        --remove_first_paragraph \
        --remove_email_headers \
        --remove_email_history \
        --remove_writes_lines
done
python -m pylowl.proj.brightside.preproc.tng_set_class_to_has_gpe data/txt/tng.concrete.orig.split
python -m pylowl.proj.brightside.preproc.tng_set_attr_to_category data/txt/tng.concrete.orig.split user
python -m pylowl.proj.brightside.preproc.tokenize_and_filter \
    data/txt/tng.orig.split/{train,test} \
    data/txt/tng/{train,test,vocab} \
    --idf_lb=0.001 --idf_ub=0.05 --lowercase --split_pattern '[^a-zA-Z]+'
