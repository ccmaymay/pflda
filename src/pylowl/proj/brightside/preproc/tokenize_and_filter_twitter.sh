#!/bin/bash

python -m pylowl.proj.brightside.preproc.tokenize_and_filter ~/ds2.concrete.split/{train,test}/'*' ~/ds2.concrete.split.tokenized.filtered/{train,test,vocab} --lowercase --idf_lb=0.01 --idf_ub=0.2 --char_filter_pattern '\W+(?=\w)' --char_filter_pattern '(?<=\w)\W+' --token_filter_pattern '@\w+$'
