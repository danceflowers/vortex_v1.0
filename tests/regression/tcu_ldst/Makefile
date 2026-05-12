ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := tcu_ldst

SRC_DIR := $(VORTEX_HOME)/tests/regression/$(PROJECT)
HW_NUM_THREADS ?= $(shell awk '/^#define NUM_THREADS/{print $$3; exit}' $(ROOT_DIR)/hw/VX_config.h)

SRCS := $(SRC_DIR)/main.cpp
VX_SRCS := $(SRC_DIR)/kernel.cpp

CXXFLAGS += -I$(VORTEX_HOME)/kernel/include

override CONFIGS += -DNUM_THREADS=$(HW_NUM_THREADS)

include ../common.mk
