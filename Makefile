CC = gcc
PYTHON = python

ifeq ($(shell uname -s),Darwin)
	# TODO check
	SHLIB_SUFFIX = .dylib
else
	SHLIB_SUFFIX = .so
endif

SRC_DIR = src
BLD_DIR = build

.PHONY: clean
clean:
	rm -rf $(BLD_DIR)

include $(SRC_DIR)/Makefile.in
