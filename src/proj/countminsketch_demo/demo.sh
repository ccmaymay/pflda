#!/bin/bash

CM_FILE=demo.dat
DATA_FILE=../../../data/txt/dog.txt

python -m cm.py read $DATA_FILE $CM_FILE
python -m cm.py query $CM_FILE 'no'
python -m cm.py query $CM_FILE 'dog bark'
python -m cm.py query $CM_FILE 'cat bark'
python -m cm.py query $CM_FILE 'dog clothes'
python -m cm.py query $CM_FILE 'cat clothes'
python -m cm.py query $CM_FILE 'dog everrrr'
python -m cm.py query $CM_FILE 'cat everrrr'
