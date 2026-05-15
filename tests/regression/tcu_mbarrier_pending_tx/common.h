#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <tensor_cfg.h>

#ifndef NUM_THREADS
#define NUM_THREADS 32
#endif

typedef struct {
  uint32_t grid_dim[2];
  uint32_t block_dim[2];
  uint64_t tensor_map_addr;  // DRAM address of the 128B tensor_map_t
  uint64_t out_addr;         // DRAM address of one u32 result slot
  uint32_t tx_bytes;         // expected bytes for mbarrier_expect_tx
} kernel_arg_t;

#endif
