#### MACROS ###########################

CC = gcc

CC_FLAGS = -O0 -g -std=gnu99 -Wall -fPIC
LD_FLAGS = -shared

LIB_NAME = liblowl.so
OBJECTS = lowl_hash.o lowl_math.o lowl_sketch.o

#######################################

.PHONY: all
all : ${OBJECTS} tests

%.o: %.c %.h
	$(CC) $(CC_FLAGS) -c $<

tests : tests.o $(LIB_NAME)
	${CC} ${CC_FLAGS} ${LD_FLAGS} -o tests $< $(LIB_NAME)

tests.o : tests.c lowl_math.h lowl_hash.h
	${CC} ${CC_FLAGS} -c $<

$(LIB_NAME): $(OBJECTS)
	$(CC) $(CC_FLAGS) $(LD_FLAGS) -o $(LIB_NAME) $(OBJECTS) -lm

.PHONY: clean
clean:
	rm -rf $(LIB_NAME) *.o tests
