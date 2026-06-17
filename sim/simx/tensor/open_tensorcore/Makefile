# TensorCore simulator Makefile (current codebase)

CXX        ?= g++
BUILD_DIR  ?= build

ifeq ($(OS),Windows_NT)
EXE_EXT    := .exe
TARGET     ?= $(BUILD_DIR)/main$(EXE_EXT)
RUN_BIN    := $(TARGET)
MKDIR_P    := if not exist "$(BUILD_DIR)" mkdir "$(BUILD_DIR)"
CLEAN_CMD  := if exist "$(BUILD_DIR)" rmdir /S /Q "$(BUILD_DIR)"
else
EXE_EXT    :=
TARGET     ?= $(BUILD_DIR)/main
RUN_BIN    := ./$(TARGET)
MKDIR_P    := mkdir -p "$(BUILD_DIR)"
CLEAN_CMD  := rm -rf "$(BUILD_DIR)"
endif

SRCS       := tensor_helper/test/main.cpp
OBJS       := $(SRCS:%.cpp=$(BUILD_DIR)/%.o)
DEPS       := $(OBJS:.o=.d)

HDRS := $(wildcard tensor_top/*.h) \
        $(wildcard tensor_control/*.h) \
        $(wildcard local_memory/*.h) \
        $(wildcard tensor_compute/*.h) \
        $(wildcard tensor_helper/*.h) \
        $(wildcard tensor_helper/test/*.h)

CPPFLAGS   ?= -I. -I./tensor_top -I./tensor_control -I./local_memory -I./tensor_compute -I./tensor_helper -I./tensor_helper/test -I.. -I../../../common
CXXFLAGS   ?= -std=c++17 -Wall -Wextra -Wpedantic
LDFLAGS    ?=
LDLIBS     ?=

RELEASE_FLAGS ?= -O2
DEBUG_FLAGS   ?= -O0 -g -DDEBUG

# Runtime options (override on command line)
TEST_ID  ?= 0
RM_MODE  ?= RNE
SEED     ?= 0

RUN_ARGS := --test $(TEST_ID)
ifneq ($(RM_MODE),RNE)
RUN_ARGS += --rm $(RM_MODE)
endif
ifneq ($(SEED),0)
RUN_ARGS += --seed $(SEED)
endif

.PHONY: all release debug run run-all test clean rebuild help print-config

all: release

release: CXXFLAGS += $(RELEASE_FLAGS)
release: $(TARGET)

debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR):
	$(MKDIR_P)

$(BUILD_DIR)/%.o: %.cpp $(HDRS) | $(BUILD_DIR)
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

run: release
	$(RUN_BIN) $(RUN_ARGS)

run-all: release
	$(RUN_BIN)

test: run

print-config:
	@echo CXX       = $(CXX)
	@echo TARGET    = $(TARGET)
	@echo BUILD_DIR = $(BUILD_DIR)
	@echo RUN_ARGS  = $(RUN_ARGS)

clean:
	$(CLEAN_CMD)

rebuild: clean all

help:
	@echo Usage:
	@echo   mingw32-make / mingw32-make release   Build optimized binary
	@echo   mingw32-make debug                     Build debug binary
	@echo   mingw32-make run                       Build and run with RUN_ARGS
	@echo   mingw32-make run-all                   Build and run default (all tests)
	@echo   mingw32-make test                      Alias of run
	@echo   mingw32-make clean                     Remove build outputs
	@echo.
	@echo Runtime options (override):
	@echo   TEST_ID: 0=all, 1=dense, 2=asym, 3=multi-K, 4=sparse-2:4, 5=sparse-1:4
	@echo   RM_MODE: RNE or RTZ or RDN or RUP or RMM
	@echo   SEED: unsigned integer
	@echo Example:
	@echo   mingw32-make run TEST_ID=4 RM_MODE=RNE SEED=1

-include $(DEPS)
