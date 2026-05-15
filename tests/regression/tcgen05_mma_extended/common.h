#pragma once

#include <stdint.h>

#define M_DIM 16
#define N_DIM 16
#define K_DIM 16
#define NUM_TCU_LD_WORDS 8
#define TCGEN05_PAYLOAD_BYTES 512
#define TCGEN05_C_PAYLOAD_BYTES 1024
#define TCGEN05_META_BYTES 64

#define TCGEN05_CASE_ASYM_F16_F8_F32 1
#define TCGEN05_CASE_SPARSE_2_4_F8   2
#define TCGEN05_CASE_SPARSE_1_4_F8   3

struct kernel_arg_t {
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  uint64_t a_tmap_addr;
  uint64_t b_tmap_addr;
  uint64_t c_tmap_addr;
  uint64_t meta_tmap_addr;
  uint64_t out_addr;
  uint32_t case_id;
};
