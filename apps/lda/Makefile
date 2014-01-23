lda.so: setup.py lda.pyx
	CFLAGS="-I. -I../../core" LDFLAGS="-L. -Wl,-rpath -Wl,." python setup.py build_ext --inplace

.PHONY: clean
clean:
	rm -rf *.so lda.c *.pyc build
