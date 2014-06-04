CC = gcc
PYTHON = python

ifeq ($(shell uname -s),Darwin)
	# TODO check
	SHLIB_SUFFIX = .dylib
	SHLIB_NAME_FLAG = -install_name
else
	SHLIB_SUFFIX = .so
	SHLIB_NAME_FLAG = -soname
endif

SRC_DIR = src
BLD_DIR = build

.PHONY: clean
clean:
	rm -rf $(BLD_DIR)

include $(SRC_DIR)/Makefile.in
