#!/bin/bash
python data.py transform_tng ../../../data/txt/20news-bydate-{train,test} ../../../data/txt/tng-nonalpha --split_mode=nonalpha --stop_list_path=../../../data/txt/stop_list.txt --lower=t
python data.py transform_tng ../../../data/txt/20news-bydate-{train,test} ../../../data/txt/tng-nonalpha-mc --split_mode=nonalpha --stop_list_path=../../../data/txt/stop_list.txt
python data.py transform_tng ../../../data/txt/20news-bydate-{train,test} ../../../data/txt/tng-whitespace --split_mode=whitespace --stop_list_path=../../../data/txt/stop_list.txt --lower=t
python data.py transform_tng ../../../data/txt/20news-bydate-{train,test} ../../../data/txt/tng-whitespace-mc --split_mode=whitespace --stop_list_path=../../../data/txt/stop_list.txt
python data.py transform_tng ../../../data/txt/20news-bydate-{train,test} ../../../data/txt/tng-noheader-nonalpha --split_mode=nonalpha --stop_list_path=../../../data/txt/stop_list.txt --lower=t --remove_header=t
python data.py transform_tng ../../../data/txt/20news-bydate-{train,test} ../../../data/txt/tng-noheader-nonalpha-mc --split_mode=nonalpha --stop_list_path=../../../data/txt/stop_list.txt --remove_header=t
python data.py transform_tng ../../../data/txt/20news-bydate-{train,test} ../../../data/txt/tng-noheader-whitespace --split_mode=whitespace --stop_list_path=../../../data/txt/stop_list.txt --lower=t --remove_header=t
python data.py transform_tng ../../../data/txt/20news-bydate-{train,test} ../../../data/txt/tng-noheader-whitespace-mc --split_mode=whitespace --stop_list_path=../../../data/txt/stop_list.txt --remove_header=t
python data.py transform_tng ../../../data/txt/20news-bydate-{train,test} ../../../data/txt/tng-noheader-nowalls-nonalpha --split_mode=nonalpha --stop_list_path=../../../data/txt/stop_list.txt --lower=t --remove_header=t --remove_walls=t
python data.py transform_tng ../../../data/txt/20news-bydate-{train,test} ../../../data/txt/tng-noheader-nowalls-nonalpha-mc --split_mode=nonalpha --stop_list_path=../../../data/txt/stop_list.txt --remove_header=t --remove_walls=t
python data.py transform_tng ../../../data/txt/20news-bydate-{train,test} ../../../data/txt/tng-noheader-nowalls-whitespace --split_mode=whitespace --stop_list_path=../../../data/txt/stop_list.txt --lower=t --remove_header=t --remove_walls=t
python data.py transform_tng ../../../data/txt/20news-bydate-{train,test} ../../../data/txt/tng-noheader-nowalls-whitespace-mc --split_mode=whitespace --stop_list_path=../../../data/txt/stop_list.txt --remove_header=t --remove_walls=t
