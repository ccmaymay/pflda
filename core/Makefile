#### MACROS ###########################

PLATFORM = $(shell uname)

CC = gcc
LD = gcc
PYTHON = python

CC_FLAGS = -O0 -g -std=gnu99 -Wall -Wextra

LIB_NAME = lowl
LIB_FILENAME = lib$(LIB_NAME).so
ifeq ($(PLATFORM), Linux)
	LIB_FILENAME_FLAG = -soname
else ifeq ($(PLATFORM), Darwin)
	LIB_FILENAME_FLAG = -install_name
endif
PY_LIB_NAME = py$(LIB_NAME)
PY_LIB_FILENAME = $(PY_LIB_NAME).so
OBJECTS = lowl_types.o lowl_hash.o lowl_math.o lowl_sketch.o lowl_sample.o lowl_vectors.o
APPS_DIR = ../apps
CYTHON_VERSION = Cython-0.19.2

#######################################

.PHONY: all
all: $(OBJECTS) $(PY_LIB_FILENAME) $(LIB_FILENAME) tests

.PHONY: apps
apps: install
	cd $(APPS_DIR) && make

.PHONY: install
install: $(LIB_FILENAME) $(PY_LIB_FILENAME)
	install -m 0755 $^ $(APPS_DIR)

%.o: %.c %.h
	$(CC) $(CC_FLAGS) -fPIC -c $<

tests: tests.c lowl_math.h lowl_hash.h $(LIB_FILENAME)
	$(CC) $(CC_FLAGS) $< -Wl,-rpath,. -L. -l$(LIB_NAME) -lm -o $@

.PHONY: test
test: tests $(PY_LIB_FILENAME)
	@echo "################################################################################"
	@echo "#                                                                              #"
	@echo "#                             core tests                                       #"
	@echo "#                                                                              #"
	@echo "################################################################################"
	./tests
	@echo
	@echo "################################################################################"
	@echo "#                                                                              #"
	@echo "#                             bridge tests                                     #"
	@echo "#                                                                              #"
	@echo "################################################################################"
	$(PYTHON) -m doctest --doctest-extension=pyx -v $(PY_LIB_NAME).pyx
	@echo

$(LIB_FILENAME): $(OBJECTS)
	$(LD) -shared -Wl,-rpath,. -Wl,$(LIB_FILENAME_FLAG),$(LIB_FILENAME) -lm -o $@ $(OBJECTS)

$(PY_LIB_FILENAME): setup.py $(PY_LIB_NAME).pyx $(LIB_NAME).pxd $(LIB_FILENAME)
	CFLAGS="-I." LDFLAGS="-L. -Wl,-rpath -Wl,." $(PYTHON) setup.py build_ext --inplace


## bvd: this does not work for everyone, but these are the commands that you can
## run yourself to get a user local version of Cython installed, if you can't
## otherwise have it installed system-wide. After the following is complete, you
## can delete the Cython directory, is it no longer needed (all built libraries
## are installed under ~/.local)
.PHONY: cython
cython:
	wget http://cython.org/release/$(CYTHON_VERSION).tar.gz
	tar -zxvf $(CYTHON_VERSION).tar.gz
	cd $(CYTHON_VERSION)
	$(PYTHON) setup.py install --user

.PHONY: clean
clean:
	rm -rf *.dSYM *.so *.o tests $(PY_LIB_NAME).c build
