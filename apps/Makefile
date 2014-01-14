BF_FILE = x
DATA_FILE = ../data/txt/dog.txt

.PHONY: run
run: bf.py
	python -m bf read $(DATA_FILE) $(BF_FILE)
	python -m bf query $(DATA_FILE) $(BF_FILE) 'no'
	python -m bf query $(DATA_FILE) $(BF_FILE) 'dog bark'
	python -m bf query $(DATA_FILE) $(BF_FILE) 'cat bark'
	python -m bf query $(DATA_FILE) $(BF_FILE) 'dog clothes'
	python -m bf query $(DATA_FILE) $(BF_FILE) 'cat clothes'
	python -m bf query $(DATA_FILE) $(BF_FILE) 'dog everrrr'
	python -m bf query $(DATA_FILE) $(BF_FILE) 'cat everrrr'
