#### MACROS ###########################

PLATFORM = $(shell uname)

CC = gcc
LD = gcc

CC_FLAGS = -O0 -g -std=gnu99 -Wall

LIB_NAME = lowl
LIB_FILENAME = lib$(LIB_NAME).so
ifeq ($(PLATFORM), Linux)
	LIB_FILENAME_FLAG = -soname
else ifeq ($(PLATFORM), Darwin)
	LIB_FILENAME_FLAG = -install_name
endif
PY_LIB_NAME = py$(LIB_NAME)
PY_LIB_FILENAME = $(PY_LIB_NAME).so
OBJECTS = lowl_hash.o lowl_math.o lowl_sketch.o lowl_bloom.o
APPS_DIR = ../apps
CYTHON_VERSION = Cython-0.19.2

#######################################

.PHONY: all
all: $(OBJECTS) $(PY_LIB_FILENAME) $(LIB_FILENAME) tests

.PHONY: install
install: $(LIB_FILENAME) $(PY_LIB_FILENAME)
	install -m 0755 $^ $(APPS_DIR)

%.o: %.c %.h
	$(CC) $(CC_FLAGS) -fPIC -c $<

tests: tests.c lowl_math.h lowl_hash.h $(LIB_FILENAME)
	$(CC) $(CC_FLAGS) $< -Wl,-rpath,. -L. -l$(LIB_NAME) -lm -o $@

$(LIB_FILENAME): $(OBJECTS)
	$(LD) -shared -Wl,-rpath,. -Wl,$(LIB_FILENAME_FLAG),$(LIB_FILENAME) -lm -o $@ $(OBJECTS)

$(PY_LIB_FILENAME): setup.py $(PY_LIB_NAME).pyx $(LIB_NAME).pxd $(LIB_FILENAME)
	CFLAGS="-I." LDFLAGS="-L. -Wl,-rpath -Wl,." python setup.py build_ext --inplace

.PHONY: cython
cython:
	wget http://cython.org/release/$(CYTHON_VERSION).tar.gz
	cd $(CYTHON_VERSION)
	python setup.py install --user

.PHONY: clean
clean:
	rm -rf $(LIB_NAME) *.o tests $(PY_LIB_NAME).c build
