#!/bin/bash
cd ..
python -m lda.lda make_vocab vocab.dat ../data/txt/lda_input.txt 
python -m lda.lda run_lda vocab.dat ../data/txt/lda_input.txt 
