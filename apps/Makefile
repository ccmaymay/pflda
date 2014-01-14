BF_FILE = bf.dat
DATA_FILE = ../data/txt/dog.txt

.PHONY: run
run: bf.py
	python -m bf read $(DATA_FILE) $(BF_FILE)
	python -m bf query $(BF_FILE) 'no'
	python -m bf query $(BF_FILE) 'dog bark'
	python -m bf query $(BF_FILE) 'cat bark'
	python -m bf query $(BF_FILE) 'dog clothes'
	python -m bf query $(BF_FILE) 'cat clothes'
	python -m bf query $(BF_FILE) 'dog everrrr'
	python -m bf query $(BF_FILE) 'cat everrrr'
