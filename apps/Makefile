BF_FILE = bf.dat
RS_FILE = rs.dat
DATA_FILE = ../data/txt/dog.txt

.PHONY: run
run: bf rs

.PHONY: bf
bf: bf.py
	python -m $@ read $(DATA_FILE) $(BF_FILE)
	python -m $@ query $(BF_FILE) 'no'
	python -m $@ query $(BF_FILE) 'dog bark'
	python -m $@ query $(BF_FILE) 'cat bark'
	python -m $@ query $(BF_FILE) 'dog clothes'
	python -m $@ query $(BF_FILE) 'cat clothes'
	python -m $@ query $(BF_FILE) 'dog everrrr'
	python -m $@ query $(BF_FILE) 'cat everrrr'

.PHONY: rs
rs: rs.py
	python -m $@ read $(DATA_FILE) $(RS_FILE)
	python -m $@ query $(RS_FILE) 'no'
	python -m $@ query $(RS_FILE) 'dog bark'
	python -m $@ query $(RS_FILE) 'cat bark'
	python -m $@ query $(RS_FILE) 'dog clothes'
	python -m $@ query $(RS_FILE) 'cat clothes'
	python -m $@ query $(RS_FILE) 'dog everrrr'
	python -m $@ query $(RS_FILE) 'cat everrrr'
