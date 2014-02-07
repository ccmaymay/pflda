#!/bin/bash
python data.py transform_tng ../../data/txt/20news-bydate-{train,test} ../../data/txt/tng-nonalpha --split_mode=nonalpha --stop_list_path=../../data/txt/stop_list.txt --lower=t
python data.py transform_tng ../../data/txt/20news-bydate-{train,test} ../../data/txt/tng-whitespace --split_mode=whitespace --stop_list_path=../../data/txt/stop_list.txt --lower=t
python data.py transform_tng ../../data/txt/20news-bydate-{train,test} ../../data/txt/tng-whitespace-mc --split_mode=whitespace --stop_list_path=../../data/txt/stop_list.txt
