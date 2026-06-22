ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tcgen05_mma_extended

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)
HW_NUM_THREADS ?= 32
SRCS := $(SRC_DIR)/main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp
VX_SRCS := $(SRC_DIR)/kernel.cpp

CXXFLAGS += -I$(VORTEX_HOME)/kernel/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -I$(VORTEX_HOME)/sim/simx

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

override CONFIGS += -DNUM_THREADS=$(HW_NUM_THREADS)

include ../common.mk



run:

LD_LIBRARY_PATH=$(VORTEX_HOME)/runtime:$(VORTEX_HOME)/sim/simx:$$LD_LIBRARY_PATH ./tcgen05_mma_extended



run-simx: run

