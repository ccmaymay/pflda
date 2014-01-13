#### MACROS ###########################

CC = gcc
LD = gcc

CC_FLAGS = -O0 -g -std=gnu99 -Wall

LIB_NAME = lowl
LIB_SONAME = lib$(LIB_NAME).so
PY_LIB_FILENAME = $(LIB_NAME).so
OBJECTS = lowl_hash.o lowl_math.o lowl_sketch.o lowl_table.o

#######################################

.PHONY: all
all: $(OBJECTS) tests

%.o: %.c %.h
	$(CC) $(CC_FLAGS) -fPIC -c $<

tests: tests.c lowl_math.h lowl_hash.h $(LIB_SONAME)
	$(CC) $(CC_FLAGS) $< -Wl,-rpath,. -L. -l$(LIB_NAME) -lm -o $@

$(LIB_SONAME): $(OBJECTS)
	$(LD) -shared -Wl,-rpath,. -Wl,-soname,$(LIB_SONAME) -lm -o $@ $(OBJECTS)

.PHONY: clean
clean:
	rm -rf $(LIB_NAME) *.o tests

pylowl.so: setup.py pylowl.pyx lowl.pxd $(LIB_SONAME)
	CFLAGS="-I." LDFLAGS="-L. -Wl,-rpath -Wl,." python setup.py build_ext --inplace
