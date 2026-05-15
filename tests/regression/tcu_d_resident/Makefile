ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tcu_d_resident

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)
HW_NUM_THREADS ?= $(shell awk '/^#define NUM_THREADS/{print $$3; exit}' $(ROOT_DIR)/hw/VX_config.h)

SRCS := $(SRC_DIR)/main.cpp $(SW_COMMON_DIR)/rvfloats.cpp $(SW_COMMON_DIR)/softfloat_ext.cpp
VX_SRCS := $(SRC_DIR)/kernel.cpp

CXXFLAGS += -I$(VORTEX_HOME)/kernel/include
CXXFLAGS += -I$(THIRD_PARTY_DIR)/softfloat/source/include
CXXFLAGS += -I$(VORTEX_HOME)/sim/simx

LDFLAGS += $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a

override CONFIGS += -DNUM_THREADS=$(HW_NUM_THREADS)

include ../common.mk
