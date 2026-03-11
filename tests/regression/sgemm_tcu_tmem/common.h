#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef NUM_THREADS
#define NUM_THREADS 32
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

typedef struct {
  uint32_t block_dim[2];
  uint32_t tile_grid[2];
  uint32_t bank_span;
  uint64_t in_desc_addr;
  uint64_t out_desc_addr;
} kernel_arg_t;

typedef struct __attribute__((packed)) {
  uint64_t addr;
  uint32_t size_bytes;
  uint32_t stride_bytes;
  uint16_t rows;
  uint16_t cols;
  uint16_t elem_bytes;
  uint16_t flags;
} tma_descriptor_t;

#endif
