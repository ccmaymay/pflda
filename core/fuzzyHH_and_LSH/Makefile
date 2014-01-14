#### MACROS ###########################

CC = gcc

OPT = -O4 -g -std=c99 -Wall

#######################################

all : util.o reservoir.o distances.o lshforest.o fuzzyhh run_tests print_points

util.o : util.c util.h Makefile
	gcc ${OPT} -c util.c

reservoir.o : reservoir.c reservoir.h Makefile
	gcc ${OPT} -c reservoir.c

distances.o : distances.c distances.h Makefile
	gcc ${OPT} -c distances.c -lm

lshforest.o : lshforest.c lshforest.h Makefile lshtree.o lshnode.o linkedlist.o
	gcc ${OPT} -c lshforest.c

lshtree.o : lshtree.c lshtree.h Makefile lshnode.o
	gcc ${OPT} -c lshtree.c

lshnode.o : lshnode.c lshnode.h Makefile
	gcc ${OPT} -c lshnode.c

linkedlist.o : linkedlist.c linkedlist.h Makefile
	gcc ${OPT} -c linkedlist.c

fuzzyhh : fuzzyhh.c reservoir.o util.o distances.o lshforest.o Makefile
	gcc ${OPT} -o fuzzyhh fuzzyhh.c reservoir.c distances.c util.c -lm

run_tests : run_tests.c reservoir.o util.o distances.o lshforest.o Makefile
	gcc ${OPT} -o run_tests run_tests.c reservoir.c distances.c util.c -lm

print_points : print_points.c util.o distances.o Makefile
	gcc ${OPT} -o print_points print_points.c distances.c util.c -lm

clean:
	rm -rf *.o fuzzyhh run_tests print_points
