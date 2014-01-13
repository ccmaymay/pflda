#### MACROS ###########################

CC = gcc
LD = gcc

CC_FLAGS = -O0 -g -std=gnu99 -Wall

LIB_NAME = lowl
LIB_SONAME = lib$(LIB_NAME).so
PY_LIB_NAME = py$(LIB_NAME)
PY_LIB_FILENAME = $(PY_LIB_NAME).so
OBJECTS = lowl_hash.o lowl_math.o lowl_sketch.o lowl_table.o

#######################################

.PHONY: all
all: $(OBJECTS) $(PY_LIB_FILENAME) $(LIB_SONAME) tests

%.o: %.c %.h
	$(CC) $(CC_FLAGS) -fPIC -c $<

tests: tests.c lowl_math.h lowl_hash.h $(LIB_SONAME)
	$(CC) $(CC_FLAGS) $< -Wl,-rpath,. -L. -l$(LIB_NAME) -lm -o $@

$(LIB_SONAME): $(OBJECTS)
	$(LD) -shared -Wl,-rpath,. -Wl,-soname,$(LIB_SONAME) -lm -o $@ $(OBJECTS)

$(PY_LIB_FILENAME): setup.py $(PY_LIB_NAME).pyx $(LIB_NAME).pxd $(LIB_SONAME)
	CFLAGS="-I." LDFLAGS="-L. -Wl,-rpath -Wl,." python setup.py build_ext --inplace

.PHONY: clean
clean:
	rm -rf $(LIB_NAME) *.o tests $(PY_LIB_NAME).c
