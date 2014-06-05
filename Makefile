.PHONY: help
help: initial-help src-help
	@echo 'Global command-line options:'
	@echo '  DEBUG:           not implemented'
	@echo '  INSTALL_PREFIX:  path to installation base directory'
	@echo '  INSTALL_USER:    boolean: if defined, use user installation scheme'
	@echo '                   in distutils (Python) and set installation base'
	@echo '                   directory to home for non-distutils targets'
	@echo '                   (e.g., lib$(LOWL_LIB_SHORTNAME))'
	@echo '                   (overrides INSTALL_PREFIX)'
	@echo
	@echo 'Hoot.'

.PHONY: initial-help
initial-help:
	@echo 'Build process occurs entirely in "$(BLD_DIR)".'
	@echo 'Sources are copied from "$(SRC_DIR)" to "$(BLD_DIR)" as needed.'
	@echo
	@echo 'Global targets:'
	@echo '  clean:  remove all files in build directory'
	@echo

PYTHON := python

CC := gcc
LD := gcc

CFLAGS += -std=gnu99 -g -O0 -Wall -Wextra
LDFLAGS +=

ifdef INSTALL_USER
	DISTUTILS_INSTALL_FLAGS += --user
	INSTALL_PREFIX := $(HOME)
else ifdef INSTALL_PREFIX
	DISTUTILS_INSTALL_FLAGS += --prefix=$(INSTALL_PREFIX)
else
	INSTALL_PREFIX := /
endif

ifeq ($(shell uname -s),Darwin)
	# TODO check
	SHLIB_SUFFIX := .dylib
	SHLIB_NAME_FLAG := -install_name
	ARCHFLAGS += -Wno-error=unused-command-line-argument-hard-error-in-future
else
	SHLIB_SUFFIX := .so
	SHLIB_NAME_FLAG := -soname
endif

INCLUDE_MAKEFILE_FILENAME := Makefile.in

SRC_DIR := src
BLD_DIR := build

ALL_SRC_FILES := $(shell find $(SRC_DIR) -type f -not -name $(INCLUDE_MAKEFILE_FILENAME))
ALL_SRC_BLD_FILES := $(patsubst $(SRC_DIR)/%,$(BLD_DIR)/%,$(ALL_SRC_FILES))

.PHONY: clean
clean:
	-@find build -type f -delete

$(ALL_SRC_BLD_FILES): $(BLD_DIR)/%: $(SRC_DIR)/%
	@mkdir -p $(@D)
	@cp -dp $< $@

include $(SRC_DIR)/$(INCLUDE_MAKEFILE_FILENAME)
