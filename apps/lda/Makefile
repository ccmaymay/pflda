PYTHON = python

lda.so: setup.py lda.pyx
	CFLAGS="-I. -I../../core -I/export/apps/lib/python2.7/site-packages/numpy/core/include" LDFLAGS="-L. -Wl,-rpath -Wl,." $(PYTHON) setup.py build_ext --inplace

.PHONY: clean
clean:
	rm -rf *.so lda.c *.pyc build
