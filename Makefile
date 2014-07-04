.PHONY: first
first: lowl

.PHONY: help
help:
	@echo 'This Makefile defines an isolated build process'
	@echo 'for the littleowl C library.'
	@echo
	@echo 'Build process occurs entirely in "$(BUILD_DIR)".'
	@echo 'Sources are copied to "$(BUILD_DIR)" as needed.'
	@echo
	@echo 'Targets:'
	@echo '  lowl:     build core C library (lib$(LIB_SHORTNAME))'
	@echo '  tests:    build and run lib$(LIB_SHORTNAME) tests'
	@echo '  install:  install lib$(LIB_SHORTNAME)'
	@echo '  clean:    remove "$(BUILD_DIR)"'
	@echo
	@echo 'Command-line options:'
	@echo '  DEBUG:    not implemented'
	@echo '  PREFIX:   path to installation prefix directory'
	@echo
	@echo 'Hoot.'

SOURCE_DIR := src/lowl
BUILD_DIR := build/lowl

CC := gcc
LD := gcc

CFLAGS += -std=gnu99 -g -O0 -Wall -Wextra
LDFLAGS +=

ifeq ($(shell uname -s),Darwin)
	SHLIB_LIB_PATH_ENV_VAR := DYLD_LIBRARY_PATH
	SHLIB_SUFFIX := .dylib
	SHLIB_NAME_FLAG := -install_name
else
	SHLIB_LIB_PATH_ENV_VAR := LD_LIBRARY_PATH
	SHLIB_SUFFIX := .so
	SHLIB_NAME_FLAG := -soname
endif

LIB_SHORTNAME := lowl
LIB_NAME := lib$(LIB_SHORTNAME)$(SHLIB_SUFFIX)
LIB_FILENAME := $(LIB_NAME)
LIB_PATH := $(BUILD_DIR)/$(LIB_FILENAME)

TESTS_OBJECT := $(BUILD_DIR)/tests
TESTS_SOURCE := $(SOURCE_DIR)/tests.c

SOURCES := $(filter-out $(TESTS_SOURCE),$(wildcard $(SOURCE_DIR)/*.c))
HEADERS := $(wildcard *.h)
OBJECTS := $(patsubst $(SOURCE_DIR)/%.c,$(BUILD_DIR)/%.o,$(SOURCES))

$(BUILD_DIR)/%.o: $(SOURCE_DIR)/%.c $(HEADERS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -fPIC -o $@ -c $<

$(LIB_PATH): $(OBJECTS)
	@mkdir -p $(@D)
	$(LD) $(LDFLAGS) -shared -Wl,$(SHLIB_NAME_FLAG),$(LIB_NAME) -o $@ $^ -lm

$(TESTS_OBJECT): $(TESTS_SOURCE) $(LIB_PATH)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ $< -L$(BUILD_DIR) -l$(LIB_SHORTNAME) -lm

.PHONY: lowl
lowl: $(LIB_PATH)

.PHONY: install
install: $(LIB_PATH)
	mkdir -p $(PREFIX)/lib
	install -m 0755 $(LIB_PATH) $(PREFIX)/lib/

.PHONY: test
test: $(TESTS_OBJECT)
	$(SHLIB_LIB_PATH_ENV_VAR)=$(BUILD_DIR):$$$(SHLIB_LIB_PATH_ENV_VAR) $(TESTS_OBJECT)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
