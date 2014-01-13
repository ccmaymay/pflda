#### MACROS ###########################

CC = gcc

OPT = -O4 -g -std=gnu99 -Wall

#######################################

all : lowl_hash.o lowl_math.o lowl_sketch.o tests

lowl_hash.o : lowl_hash.c lowl_hash.h Makefile
	${CC} ${OPT} -c lowl_hash.c -lm

lowl_math.o : lowl_math.c lowl_math.h Makefile
	gcc ${OPT} -c lowl_math.c

lowl_sketch.o : lowl_sketch.c lowl_sketch.h Makefile
	gcc ${OPT} -c lowl_sketch.c

tests : tests.o lowl_math.o lowl_hash.o
	gcc ${OPT} -o tests tests.o lowl_math.o lowl_hash.o -lm

tests.o : tests.c lowl_math.h lowl_hash.h
	gcc ${OPT} -c tests.c -lm

clean:
	rm -rf *.o tests
