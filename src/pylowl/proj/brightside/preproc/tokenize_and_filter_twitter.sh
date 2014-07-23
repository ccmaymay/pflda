#!/bin/bash

python -m pylowl.proj.brightside.preproc.tokenize_and_filter ~/ds2.concrete.split/{train,test}/'*' ~/ds2.concrete.split.tokenized.filtered/{train,test,vocab} --lowercase --idf_lb=0.0005 --idf_ub=0.1 --char_filter_pattern '\W+(?=\w)' --char_filter_pattern '(?<=\w)\W+' --token_filter_pattern '@\w+' --token_filter_pattern '.*USER_[a-z0-9]{7}'
