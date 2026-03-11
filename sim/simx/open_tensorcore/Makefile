# TensorCore simulator Makefile (current codebase)

CXX        ?= g++
BUILD_DIR  ?= build

ifeq ($(OS),Windows_NT)
EXE_EXT    := .exe
TARGET     ?= main$(EXE_EXT)
RUN_BIN    := $(TARGET)
MKDIR_P    := if not exist "$(BUILD_DIR)" mkdir "$(BUILD_DIR)"
CLEAN_CMD  := if exist "$(BUILD_DIR)" rmdir /S /Q "$(BUILD_DIR)" && if exist "$(TARGET)" del /Q "$(TARGET)"
else
EXE_EXT    :=
TARGET     ?= main
RUN_BIN    := ./$(TARGET)
MKDIR_P    := mkdir -p "$(BUILD_DIR)"
CLEAN_CMD  := rm -rf "$(BUILD_DIR)" "$(TARGET)"
endif

SRCS       := main.cpp
OBJS       := $(SRCS:%.cpp=$(BUILD_DIR)/%.o)
DEPS       := $(OBJS:.o=.d)

# Core headers in this project (for quick visibility)
HDRS := \
  config_register.h con_to_fp9.h fadd_s1.h fadd_s2.h \
  fmul_s1.h fmul_s2.h fmul_s3.h fp22_to_fp16.h fp_types.h \
  tc_add_pipe.h tc_mul_add.h tc_mul_pipe.h tensor_core_top.h

CPPFLAGS   ?=
CXXFLAGS   ?= -std=c++17 -Wall -Wextra -Wpedantic
LDFLAGS    ?=
LDLIBS     ?=

RELEASE_FLAGS ?= -O2
DEBUG_FLAGS   ?= -O0 -g -DDEBUG

# Runtime options (override on command line)
PREC     ?= FP8_E5M2
TEST_ID  ?= 2
RM_MODE  ?= RNE
SEED     ?= 0

RUN_ARGS := --prec $(PREC) --test $(TEST_ID)
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
	@echo   PREC: FP4_E2M1 or FP8_E4M3 or FP8_E5M2 or FP16 or FP32
	@echo   TEST_ID: integer
	@echo   RM_MODE: RNE or RTZ or RDN or RUP or RMM
	@echo   SEED: unsigned integer
	@echo Example:
	@echo   mingw32-make run PREC=FP16 TEST_ID=2 RM_MODE=RNE

-include $(DEPS)