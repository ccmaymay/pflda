#!/bin/bash
python data.py transform_tng ../../data/txt/20news-bydate-{train,test} ../../data/txt/tng-nonalpha nonalpha ../../data/txt/stop_list.txt
python data.py transform_tng ../../data/txt/20news-bydate-{train,test} ../../data/txt/tng-whitespace whitespace ../../data/txt/stop_list.txt
