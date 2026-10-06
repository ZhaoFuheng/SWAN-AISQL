PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=aisql
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Local `make tidy-check` on macOS: the tidy build records /usr/bin/c++ without a sysroot, and a Homebrew
# clang-tidy then finds no standard headers (every TU: "'memory' file not found", and classes with unique_ptr
# members look trivially destructible). Give that build the SDK explicitly, and default to the Homebrew binary.
# Target-specific, so the ordinary builds keep their flags (and their incremental state).
ifeq ($(shell uname -s),Darwin)
tidy-check: EXT_FLAGS += -DCMAKE_OSX_SYSROOT=$(shell xcrun --show-sdk-path)
ifeq ($(TIDY_BINARY),)
ifneq ($(wildcard /opt/homebrew/opt/llvm/bin/clang-tidy),)
TIDY_BINARY := /opt/homebrew/opt/llvm/bin/clang-tidy
else ifneq ($(wildcard /usr/local/opt/llvm/bin/clang-tidy),)
TIDY_BINARY := /usr/local/opt/llvm/bin/clang-tidy
endif
endif
TIDY_THREADS ?= 8
endif

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile
