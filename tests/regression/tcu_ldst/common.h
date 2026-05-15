#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef NUM_THREADS
#define NUM_THREADS 32
#endif

typedef struct {
  uint32_t grid_dim[2];
  uint32_t block_dim[2];
  uint64_t out_addr;     // DRAM buffer for the host to verify (NUM_THREADS u32)
} kernel_arg_t;

#endif
