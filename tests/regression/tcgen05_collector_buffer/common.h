#pragma once

#include <stdint.h>

#define M_DIM 16
#define N_DIM 16
#define K_DIM 16
#define NUM_TCU_LD_WORDS 8
#define COLLECTOR_RESULT_COUNT 1
#define TCGEN05_PAYLOAD_BYTES 512

struct kernel_arg_t {
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  uint64_t a_tmap_addr;
  uint64_t a_poison_tmap_addr;
  uint64_t b_tmap_addr;
  uint64_t out_addr;
};
