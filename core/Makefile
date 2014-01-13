#### MACROS ###########################

CC = gcc
LD = gcc

CC_FLAGS = -O0 -g -std=gnu99 -Wall -fPIC

LIB_NAME = liblowl.so
OBJECTS = lowl_hash.o lowl_math.o lowl_sketch.o

#######################################

.PHONY: all
all : $(OBJECTS) tests

%.o: %.c %.h
	$(CC) $(CC_FLAGS) -c $<

tests : tests.c lowl_math.h lowl_hash.h $(LIB_NAME)
	$(CC) $(CC_FLAGS) $< -Wl,-rpath,. -L. -llowl -lm -o $@
	#$(CC) -Wall -g -std=gnu99 -O0 -L. $< -llowl -o $@

$(LIB_NAME): $(OBJECTS)
	$(LD) -shared -Wl,-rpath,. -Wl,-soname,$(LIB_NAME) -lm -o $@ $(OBJECTS)

.PHONY: clean
clean:
	rm -rf $(LIB_NAME) *.o tests

#table.so: setup.py table.pyx stringTable.pxd libstringTable.so
#	CFLAGS="-I." LDFLAGS="-L. -Wl,-rpath -Wl,." python setup.py build_ext --inplace

#.PHONY: run
#run: table.so
#	python -m tableTest
