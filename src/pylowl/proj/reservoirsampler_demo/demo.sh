#!/bin/bash

RS_FILE=demo.dat
RS_VALUES_FILE=demo_values.dat
DATA_FILE=../../../data/txt/dog.txt

python -m rs read $DATA_FILE $RS_FILE $RS_VALUES_FILE
python -m rs sample $RS_FILE $RS_VALUES_FILE
