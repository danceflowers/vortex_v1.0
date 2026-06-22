include ../common.mk

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

DESTDIR ?= $(CURDIR)

OBJ_DIR     := $(DESTDIR)/obj
CONFIG_FILE := $(DESTDIR)/simx_config.stamp

# Force canonical paths for this Makefile.

# When running `make -C sim/simx`, CURDIR is /workspace/vortex/sim/simx.

# Therefore:

#   ROOT_DIR      = /workspace/vortex

#   SRC_DIR       = /workspace/vortex/sim/simx

#   SW_COMMON_DIR = /workspace/vortex/sim/common

override ROOT_DIR := $(abspath $(CURDIR)/../..)

override VORTEX_HOME := $(ROOT_DIR)

override SRC_DIR := $(CURDIR)

override SW_COMMON_DIR := $(ROOT_DIR)/sim/common

# common.mk/config.mk normally defines VORTEX_HOME.
# This fallback prevents SRC_DIR from becoming /sim/simx if VORTEX_HOME is empty.
VORTEX_HOME ?= $(ROOT_DIR)

SRC_DIR ?= $(VORTEX_HOME)/sim/simx

# Fallback for common software directory if config.mk does not define it.
SW_COMMON_DIR ?= $(VORTEX_HOME)/sim/common

# -------------------------------------------------------------------
# Compiler flags
# -------------------------------------------------------------------

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -fPIC -Wno-maybe-uninitialized

CXXFLAGS += -I$(SRC_DIR)
CXXFLAGS += -I$(SRC_DIR)/common
CXXFLAGS += -I$(SRC_DIR)/tensor
CXXFLAGS += -I$(SRC_DIR)/tensor/open_tensorcore
CXXFLAGS += -I$(SRC_DIR)/tensor/open_tensorcore/tensor_compute
CXXFLAGS += -I$(SRC_DIR)/tensor/open_tensorcore/tensor_control
CXXFLAGS += -I$(SRC_DIR)/tensor/open_tensorcore/tensor_helper
CXXFLAGS += -I$(SW_COMMON_DIR)
CXXFLAGS += -I$(ROOT_DIR)/hw

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/ramulator/ext/spdlog/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/ramulator/ext/yaml-cpp/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/ramulator/src

CXXFLAGS += -DXLEN_$(XLEN)
CXXFLAGS += $(CONFIGS)

# generate .d files alongside .o files
CXXFLAGS += -MMD -MP -MF $(@:.o=.d)

# optional: pipe through ccache if available
CXX := $(if $(shell which ccache),ccache $(CXX),$(CXX))

# -------------------------------------------------------------------
# Link flags
# -------------------------------------------------------------------

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a
LDFLAGS += -Wl,-rpath,$(THIRD_PARTY_DIR)/ramulator
LDFLAGS += -L$(THIRD_PARTY_DIR)/ramulator -lramulator

# -------------------------------------------------------------------
# Source files
# -------------------------------------------------------------------

SRCS :=

# shared simulator common sources
SRCS += $(SW_COMMON_DIR)/util.cpp
SRCS += $(SW_COMMON_DIR)/mem.cpp
SRCS += $(SW_COMMON_DIR)/softfloat_ext.cpp
SRCS += $(SW_COMMON_DIR)/rvfloats.cpp
SRCS += $(SW_COMMON_DIR)/dram_sim.cpp

# simx core sources
SRCS += $(SRC_DIR)/processor.cpp
SRCS += $(SRC_DIR)/cluster.cpp
SRCS += $(SRC_DIR)/socket.cpp
SRCS += $(SRC_DIR)/core.cpp
SRCS += $(SRC_DIR)/emulator.cpp
SRCS += $(SRC_DIR)/decode.cpp
SRCS += $(SRC_DIR)/opc_unit.cpp
SRCS += $(SRC_DIR)/dispatcher.cpp
SRCS += $(SRC_DIR)/execute.cpp
SRCS += $(SRC_DIR)/func_unit.cpp
SRCS += $(SRC_DIR)/cache_sim.cpp
SRCS += $(SRC_DIR)/mem_sim.cpp
SRCS += $(SRC_DIR)/local_mem.cpp
SRCS += $(SRC_DIR)/mem_coalescer.cpp
SRCS += $(SRC_DIR)/dcrs.cpp

# simx-local common source
# IMPORTANT:
# This file lives under sim/simx/common/types.cpp.
# It must generate obj/common/types.o.
SRCS += $(SRC_DIR)/common/types.cpp

# V extension sources
ifneq ($(findstring -DEXT_V_ENABLE,$(CONFIGS)),)
SRCS += $(SRC_DIR)/voperands.cpp
SRCS += $(SRC_DIR)/vopc_unit.cpp
SRCS += $(SRC_DIR)/vec_unit.cpp
else
SRCS += $(SRC_DIR)/operands.cpp
endif

# TCU / Tensor extension sources
ifneq ($(findstring -DEXT_TCU_ENABLE,$(CONFIGS)),)
SRCS += $(SRC_DIR)/tensor/tma.cpp
SRCS += $(SRC_DIR)/tensor/tmem_utils.cpp
SRCS += $(SRC_DIR)/tensor/tmem_math_packet.cpp
SRCS += $(SRC_DIR)/tensor/tmem.cpp
SRCS += $(SRC_DIR)/tensor/tmem_system.cpp
SRCS += $(SRC_DIR)/tensor/open_tensorcore/tensor_control/tc_decode.cpp
SRCS += $(SRC_DIR)/tensor/open_tensorcore/tensor_helper/tensor_debug_utils.cpp
SRCS += $(SRC_DIR)/tensor/open_tensorcore/tensor_unit.cpp
endif

# -------------------------------------------------------------------
# Debug / perf flags
# -------------------------------------------------------------------

ifdef DEBUG
CXXFLAGS += -g -O0 -DDEBUG_LEVEL=$(DEBUG)
else
CXXFLAGS += -O2 -DNDEBUG
endif

ifdef PERF
CXXFLAGS += -DPERF_ENABLE
endif

# -------------------------------------------------------------------
# Object mapping
# -------------------------------------------------------------------

COMMON_SRCS := $(filter $(SW_COMMON_DIR)/%.cpp,$(SRCS))
SRC_SRCS    := $(filter $(SRC_DIR)/%.cpp,$(SRCS))

COMMON_OBJS := $(patsubst $(SW_COMMON_DIR)/%.cpp,$(OBJ_DIR)/common/%.o,$(COMMON_SRCS))
SRC_OBJS    := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_SRCS))

OBJS     := $(COMMON_OBJS) $(SRC_OBJS)
MAIN_OBJ := $(OBJ_DIR)/main.o

DEPS := $(OBJS:.o=.d) $(MAIN_OBJ:.o=.d)

PROJECT := simx

# -------------------------------------------------------------------
# Targets
# -------------------------------------------------------------------

.PHONY: all force clean clean-lib clean-exe clean-obj print-vars

all: $(DESTDIR)/$(PROJECT)

# -------------------------------------------------------------------
# Build rules
# -------------------------------------------------------------------

# Rule 1:
# sim/common/*.cpp -> sim/simx/obj/common/*.o
$(OBJ_DIR)/common/%.o: $(SW_COMMON_DIR)/%.cpp $(CONFIG_FILE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule 2:
# Explicit rule for sim/simx/common/types.cpp.
# This fixes:
# No rule to make target '.../sim/simx/obj/common/types.o'
$(OBJ_DIR)/common/types.o: $(SRC_DIR)/common/types.cpp $(CONFIG_FILE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule 3:
# sim/simx/*.cpp and subdirectories -> sim/simx/obj/*.o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(CONFIG_FILE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# main.cpp needs STARTUP_ADDR
$(MAIN_OBJ): $(SRC_DIR)/main.cpp $(CONFIG_FILE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -DSTARTUP_ADDR=0x80000000 -c $< -o $@

# Main executable
$(DESTDIR)/$(PROJECT): $(OBJS) $(MAIN_OBJ)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Shared library
$(DESTDIR)/lib$(PROJECT).so: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -shared $(LDFLAGS) -o $@

# Update timestamp when flags changed
$(CONFIG_FILE): force
	@mkdir -p $(@D)
	@printf '%s\n' "$(CXXFLAGS)" > $@.tmp
	@if ! cmp -s $@.tmp $@; then \
		mv $@.tmp $@; \
	else \
		rm $@.tmp; \
	fi

# Debug helper
print-vars:
	@echo "ROOT_DIR        = $(ROOT_DIR)"
	@echo "VORTEX_HOME     = $(VORTEX_HOME)"
	@echo "SRC_DIR         = $(SRC_DIR)"
	@echo "SW_COMMON_DIR   = $(SW_COMMON_DIR)"
	@echo "OBJ_DIR         = $(OBJ_DIR)"
	@echo "CONFIGS         = $(CONFIGS)"
	@echo "CXX             = $(CXX)"
	@echo "XLEN            = $(XLEN)"
	@echo "OBJS:"
	@printf '  %s\n' $(OBJS)

# Include auto-generated header deps
-include $(DEPS)

clean-lib:
	rm -f $(DESTDIR)/lib$(PROJECT).so

clean-exe:
	rm -f $(DESTDIR)/$(PROJECT)

clean-obj:
	rm -rf $(OBJ_DIR)

clean: clean-lib clean-exe clean-obj
