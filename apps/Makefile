BF_FILE = x
DATA_FILE = ../data/txt/google-sample-grams.txt

.PHONY: run
run: bf.py
	python -m bf read $(DATA_FILE) $(BF_FILE)
	python -m bf query $(DATA_FILE) $(BF_FILE) 'no'
	python -m bf query $(DATA_FILE) $(BF_FILE) 'woman'
	python -m bf query $(DATA_FILE) $(BF_FILE) 'woman crossdress'
	python -m bf query $(DATA_FILE) $(BF_FILE) 'woman crossdressing'
	python -m bf query $(DATA_FILE) $(BF_FILE) 'woman tracy'
	python -m bf query $(DATA_FILE) $(BF_FILE) 'woman trade'
	python -m bf query $(DATA_FILE) $(BF_FILE) 'woman tracks'
	python -m bf query $(DATA_FILE) $(BF_FILE) 'woman track'
	python -m bf query $(DATA_FILE) $(BF_FILE) 'woman toy'
	python -m bf query $(DATA_FILE) $(BF_FILE) 'woman toying'
	python -m bf query $(DATA_FILE) $(BF_FILE) 'woman track'
	python -m bf query $(DATA_FILE) $(BF_FILE) 'woman trackasdf'
	python -m bf query $(DATA_FILE) $(BF_FILE) 'woman asdf'
	python -m bf query $(DATA_FILE) $(BF_FILE) 'woman'
