include ../common.mk

DESTDIR ?= $(CURDIR)

OBJ_DIR = $(DESTDIR)/obj
CONFIG_FILE = $(DESTDIR)/simx_config.stamp
SRC_DIR = $(VORTEX_HOME)/sim/simx

CXXFLAGS += -std=c++17 -Wall -Wextra -Wfatal-errors
CXXFLAGS += -fPIC -Wno-maybe-uninitialized
CXXFLAGS += -I$(SRC_DIR) -I$(SRC_DIR)/common -I$(SRC_DIR)/tensor
CXXFLAGS += -I$(SRC_DIR)/tensor/open_tensorcore -I$(SRC_DIR)/tensor/open_tensorcore/tensor_compute
CXXFLAGS += -I$(SRC_DIR)/tensor/open_tensorcore/local_memory -I$(SRC_DIR)/tensor/open_tensorcore/tensor_control
CXXFLAGS += -I$(SRC_DIR)/tensor/open_tensorcore/tensor_helper -I$(SRC_DIR)/tensor/open_tensorcore/tensor_top
CXXFLAGS += -I$(SW_COMMON_DIR) -I$(ROOT_DIR)/hw
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/ramulator/ext/spdlog/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/ramulator/ext/yaml-cpp/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/ramulator/src
CXXFLAGS += -DXLEN_$(XLEN)
CXXFLAGS += $(CONFIGS)

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a
LDFLAGS += -Wl,-rpath,$(THIRD_PARTY_DIR)/ramulator -L$(THIRD_PARTY_DIR)/ramulator -lramulator

# Source files definition
SRCS = $(SW_COMMON_DIR)/util.cpp $(SW_COMMON_DIR)/mem.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/dram_sim.cpp
SRCS += $(SRC_DIR)/processor.cpp $(SRC_DIR)/cluster.cpp $(SRC_DIR)/socket.cpp $(SRC_DIR)/core.cpp $(SRC_DIR)/emulator.cpp
SRCS += $(SRC_DIR)/decode.cpp $(SRC_DIR)/opc_unit.cpp $(SRC_DIR)/dispatcher.cpp
SRCS += $(SRC_DIR)/execute.cpp $(SRC_DIR)/func_unit.cpp
SRCS += $(SRC_DIR)/cache_sim.cpp $(SRC_DIR)/mem_sim.cpp $(SRC_DIR)/local_mem.cpp $(SRC_DIR)/mem_coalescer.cpp
SRCS += $(SRC_DIR)/dcrs.cpp $(SRC_DIR)/common/types.cpp

# Add V extension sources
ifneq ($(findstring -DEXT_V_ENABLE, $(CONFIGS)),)
  	SRCS += $(SRC_DIR)/voperands.cpp
  	SRCS += $(SRC_DIR)/vopc_unit.cpp
  	SRCS += $(SRC_DIR)/vec_unit.cpp
else
	SRCS += $(SRC_DIR)/operands.cpp
endif
# Add TCU extension sources
ifneq ($(findstring -DEXT_TCU_ENABLE, $(CONFIGS)),)
	SRCS += $(SRC_DIR)/tensor/tma.cpp $(SRC_DIR)/tensor/tma_frontend.cpp
	SRCS += $(SRC_DIR)/tensor/tmem.cpp $(SRC_DIR)/tensor/tmem_system.cpp $(SRC_DIR)/tensor/tmem_window_planner.cpp
	SRCS += $(SRC_DIR)/tensor/tensor_mem_layout_utils.cpp
	SRCS += $(SRC_DIR)/tensor/open_tensorcore/local_memory/tensor_local_mem_arbiter.cpp
	# tensor_local_mem_pipeline.cpp references a header that doesn't exist; skip
	SRCS += $(SRC_DIR)/tensor/open_tensorcore/local_memory/tensor_mem_pipeline.cpp
	SRCS += $(SRC_DIR)/tensor/open_tensorcore/tensor_control/tc_decode.cpp
	SRCS += $(SRC_DIR)/tensor/open_tensorcore/tensor_control/tensor_async_frontend.cpp
	SRCS += $(SRC_DIR)/tensor/open_tensorcore/tensor_control/tensor_issue_policy.cpp
	SRCS += $(SRC_DIR)/tensor/open_tensorcore/tensor_control/tensor_mem_manager.cpp
	SRCS += $(SRC_DIR)/tensor/open_tensorcore/tensor_control/tensor_wmma_issue_engine.cpp
	SRCS += $(SRC_DIR)/tensor/open_tensorcore/tensor_control/tensor_wmma_retire_unit.cpp
	SRCS += $(SRC_DIR)/tensor/open_tensorcore/tensor_helper/tensor_debug_utils.cpp
	SRCS += $(SRC_DIR)/tensor/open_tensorcore/tensor_top/tensor_runtime_utils.cpp
	SRCS += $(SRC_DIR)/tensor/open_tensorcore/tensor_top/tensor_tick_utils.cpp
	SRCS += $(SRC_DIR)/tensor/open_tensorcore/tensor_top/tensor_unit.cpp
endif

# Debugging
ifdef DEBUG
	CXXFLAGS += -g -O0 -DDEBUG_LEVEL=$(DEBUG)
else
	CXXFLAGS += -O2 -DNDEBUG
endif

# Enable perf counters
ifdef PERF
	CXXFLAGS += -DPERF_ENABLE
endif

# Convert sources to object files (in build directory)
COMMON_SRCS := $(filter $(SW_COMMON_DIR)/%.cpp,$(SRCS))
SRC_SRCS    := $(filter $(SRC_DIR)/%.cpp,$(SRCS))
COMMON_OBJS := $(patsubst $(SW_COMMON_DIR)/%.cpp,$(OBJ_DIR)/common/%.o,$(COMMON_SRCS))
SRC_OBJS    := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRC_SRCS))
OBJS        := $(COMMON_OBJS) $(SRC_OBJS)
MAIN_OBJ    := $(OBJ_DIR)/main.o

DEPS := $(OBJS:.o=.d) $(MAIN_OBJ:.o=.d)

# generate .d files alongside .o files
CXXFLAGS += -MMD -MP -MF $(@:.o=.d)

# optional: pipe through ccache if you have it
CXX := $(if $(shell which ccache),ccache $(CXX),$(CXX))

PROJECT := simx

.PHONY: all force clean clean-lib clean-exe clean-obj

all: $(DESTDIR)/$(PROJECT)

# build common object files
$(OBJ_DIR)/common/%.o: $(SW_COMMON_DIR)/%.cpp $(CONFIG_FILE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# build source object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(CONFIG_FILE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# build main object file
$(MAIN_OBJ): $(SRC_DIR)/main.cpp $(CONFIG_FILE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -DSTARTUP_ADDR=0x80000000 -c $< -o $@

# Main executable
$(DESTDIR)/$(PROJECT): $(OBJS) $(MAIN_OBJ)
	$(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@

# Shared library
$(DESTDIR)/lib$(PROJECT).so: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -shared $(LDFLAGS) -o $@

# updates the timestamp when flags changed.
$(CONFIG_FILE): force
	@mkdir -p $(@D)
	@printf '%s\n' "$(CXXFLAGS)" > $@.tmp
	@if ! cmp -s $@.tmp $@; then \
	  mv $@.tmp $@; \
	else \
	  rm $@.tmp; \
	fi

# include the auto-generated header deps; silences warnings if missing
-include $(DEPS)

clean-lib:
	rm -f $(DESTDIR)/lib$(PROJECT).so

clean-exe:
	rm -f $(DESTDIR)/$(PROJECT)

clean-obj:
	rm -rf $(OBJ_DIR)

clean: clean-lib clean-exe clean-obj
