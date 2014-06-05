PYTHON = python

CC = gcc
LD = gcc

CFLAGS += -std=gnu99 -g -O0 -Wall -Wextra
LDFLAGS +=

ifdef INSTALL_USER
	DISTUTILS_INSTALL_FLAGS = --user
else ifdef INSTALL_PREFIX
	DISTUTILS_INSTALL_FLAGS = --prefix=$(INSTALL_PREFIX)
endif

ifndef INSTALL_PREFIX
	INSTALL_PREFIX = /
endif

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

SRC_FILES =
SRC_BLD_FILES = $(patsubst $(SRC_DIR)/%,$(BLD_DIR)/%,$(SRC_FILES))

.PHONY: all
all: lowl pylowl

.PHONY: clean
clean:
	find build -type f -delete

$(SRC_BLD_FILES): $(BLD_DIR)/%: $(SRC_DIR)/%
	mkdir -p `dirname $<`
	cp -dp $< $@

include $(SRC_DIR)/Makefile.in
