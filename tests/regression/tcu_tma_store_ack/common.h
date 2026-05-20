#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <tensor_cfg.h>

#ifndef NUM_THREADS
#define NUM_THREADS 32
#endif

#define TMA_STORE_WORDS 128
#define TMA_STORE_BYTES (TMA_STORE_WORDS * 4)

typedef struct {
  uint32_t grid_dim[2];
  uint32_t block_dim[2];
  uint64_t tensor_map_addr;
  uint64_t out_addr;
} kernel_arg_t;

#endif
