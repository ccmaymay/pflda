#!/bin/bash

for t in train test
do
    python raw_filter.py ~/20news-bydate-${t} ~/20news-bydate-${t}.filtered --remove_first_paragraph=t --remove_email_headers=t --remove_walls=t --remove_emails=t --remove_email_history=t --remove_writes_lines=t
    python filter_rasp.py ~/rasp3os/scripts/rasp.sh ~/20news-bydate-${t}.filtered ~/20news-bydate-${t}.rasp
    python format_to_doc_per_line.py ~/20news-bydate-${t}.rasp ~/20news-bydate-${t}.rasp.all.ordered
    shuf -o ~/20news-bydate-${t}.rasp.all < ~/20news-bydate-${t}.rasp.all.ordered
done
python format_to_indexed.py ~/20news-bydate-{train,test}.rasp.all ~/20news-bydate-{train,test}.indexed.idf ~/20news-bydate-vocab.indexed.idf --idf_lb=0.001 --idf_ub=0.05 --remove_non_alpha=t --lowercase=t --min_word_len=3
