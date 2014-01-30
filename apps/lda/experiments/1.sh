#!/bin/bash

bash run_30_bg_tng3.sh 1-rs0 --reservoir_size=0
bash run_30_bg_tng3.sh 1-rs100 --reservoir_size=100
bash run_30_bg_tng3.sh 1-rs1k --reservoir_size=1000
bash run_30_bg_tng3.sh 1-rs10k --reservoir_size=10000
bash run_30_bg_tng3.sh 1-rs100k --reservoir_size=100000
bash run_30_bg_tng3.sh 1-rs500k --reservoir_size=500000
