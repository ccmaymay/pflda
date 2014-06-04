PYTHON = python

CC = gcc
LD = gcc

CFLAGS += -std=gnu99 -g -O0 -Wall -Wextra
LDFLAGS +=

ifeq ($(shell uname -s),Darwin)
	# TODO check
	SHLIB_SUFFIX = .dylib
	SHLIB_NAME_FLAG = -install_name
	SHLIB_PATH_ENV_VAR = DYLD_LIBRARY_PATH
	ARCHFLAGS += -Wno-error=unused-command-line-argument-hard-error-in-future
else
	SHLIB_SUFFIX = .so
	SHLIB_NAME_FLAG = -soname
	SHLIB_PATH_ENV_VAR = LD_LIBRARY_PATH
endif

SRC_DIR = src
BLD_DIR = build

.PHONY: all
all: lowl pylowl

.PHONY: clean
clean:
	rm -rf $(BLD_DIR)

include $(SRC_DIR)/Makefile.in
