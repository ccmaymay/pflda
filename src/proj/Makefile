CM_FILE = cm.dat
BF_FILE = bf.dat
RS_FILE = rs.dat
RS_VALUES_FILE = rs_values.dat
DATA_FILE = ../data/txt/dog.txt

.PHONY: run
run: bf rs cm

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
	python -m $@ read $(DATA_FILE) $(RS_FILE) $(RS_VALUES_FILE)
	python -m $@ sample $(RS_FILE) $(RS_VALUES_FILE)

.PHONY: cm
cm: cm.py
	python -m $@ read $(DATA_FILE) $(CM_FILE)
	python -m $@ query $(CM_FILE) 'no'
	python -m $@ query $(CM_FILE) 'dog bark'
	python -m $@ query $(CM_FILE) 'cat bark'
	python -m $@ query $(CM_FILE) 'dog clothes'
	python -m $@ query $(CM_FILE) 'cat clothes'
	python -m $@ query $(CM_FILE) 'dog everrrr'
	python -m $@ query $(CM_FILE) 'cat everrrr'
