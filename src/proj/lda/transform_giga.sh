#!/bin/bash
python data.py transform_gigaword /export/common/data/corpora/LDC/LDC2011T07/data/nyt_eng ../../data/txt/gigaword --split_mode=nonalpha --stop_list_path=../../data/txt/stop_list.txt --lower=t
