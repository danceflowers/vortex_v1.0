ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tcu_mma_overlap
ITYPE ?= fp8
ATYPE ?= $(ITYPE)
BTYPE ?= $(ITYPE)
OTYPE ?= fp8
SPARSE_MODE ?= 0

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)
HW_NUM_THREADS ?= $(shell awk '/^#define NUM_THREADS/{print $$3; exit}' $(ROOT_DIR)/hw/VX_config.h)

SRCS := $(SRC_DIR)/main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp
VX_SRCS := $(SRC_DIR)/kernel.cpp

CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -I$(VORTEX_HOME)/sim/simx
CXXFLAGS += -I$(VORTEX_HOME)/sim/simx/tensor
CXXFLAGS += -I$(VORTEX_HOME)/sim/simx/tensor/open_tensorcore/tensor_compute
CXXFLAGS += -I$(VORTEX_HOME)/sim/simx/tensor/open_tensorcore/tensor_control
CXXFLAGS += -I$(VORTEX_HOME)/sim/simx/tensor/open_tensorcore/local_memory
CXXFLAGS += -I$(VORTEX_HOME)/kernel/include

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

override CONFIGS += -DNUM_THREADS=$(HW_NUM_THREADS) -DITYPE=$(ITYPE) -DATYPE=$(ATYPE) -DBTYPE=$(BTYPE) -DOTYPE=$(OTYPE) -DSPARSE_MODE=$(SPARSE_MODE)

include ../common.mk
