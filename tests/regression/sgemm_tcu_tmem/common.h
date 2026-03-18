#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <tensor_cfg.h>

#ifndef NUM_THREADS
#define NUM_THREADS 32
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef ATYPE
#define ATYPE ITYPE
#endif

#ifndef BTYPE
#define BTYPE ITYPE
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

typedef vortex::tensor::descriptor_table_arg_t descriptor_table_arg_t;

typedef struct {
  descriptor_table_arg_t desc_tables;
  uint32_t block_dim[2];
  uint32_t tile_grid[2];
  uint32_t bank_span;
} kernel_arg_t;

typedef struct __attribute__((packed)) {
  uint64_t addr;
  uint32_t size_bytes;
  uint32_t stride_bytes;
  uint16_t rows;
  uint16_t cols;
  uint16_t elem_bytes;
  uint16_t flags;
  uint16_t tmem_base;
  uint16_t meta_tmem_base;
  uint16_t bank_span;
  uint16_t meta_bank_span;
  uint8_t tile_role;
  uint8_t payload_kind;
  uint8_t reserved[14];
} tma_descriptor_t;

typedef struct __attribute__((packed)) {
  uint32_t fmt_a;
  uint32_t fmt_b;
  uint32_t fmt_c;
  uint8_t ws;
  uint8_t sp;
  uint8_t sparse_mode;
  uint8_t reserved;
} mma_descriptor_t;

#endif
