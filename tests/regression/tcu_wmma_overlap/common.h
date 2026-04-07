#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <tensor_cfg.h>

#ifndef NUM_THREADS
#define NUM_THREADS 32
#endif

#ifndef ITYPE
#define ITYPE fp8
#endif

#ifndef ATYPE
#define ATYPE ITYPE
#endif

#ifndef BTYPE
#define BTYPE ITYPE
#endif

#ifndef OTYPE
#define OTYPE fp16
#endif

#ifndef SPARSE_MODE
#define SPARSE_MODE 0
#endif

// Matrix dimensions (512x512x512 GEMM: D = A * B, C=0)
#ifndef MATRIX_M
#define MATRIX_M 512
#endif

#ifndef MATRIX_N
#define MATRIX_N 512
#endif

#ifndef MATRIX_K
#define MATRIX_K 512
#endif

// Tile dimensions (each WMMA covers m16 n16 k16, i.e. 2 x k8 sub-tiles)
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// Derived tile counts
#define M_TILES (MATRIX_M / TILE_M)
#define N_TILES (MATRIX_N / TILE_N)
#define K_PHASES (MATRIX_K / TILE_K)

typedef vortex::tensor::descriptor_table_arg_t descriptor_table_arg_t;

typedef struct {
  descriptor_table_arg_t desc_tables;
  uint32_t block_dim[2];
  uint32_t a_bank_span;
  uint32_t b_bank_span;
  uint32_t c_bank_span;
} kernel_arg_t;

typedef struct __attribute__((packed)) {
  uint64_t addr;
  uint32_t size_bytes;
  uint32_t stride_bytes;
  uint16_t rows;
  uint16_t cols;
  uint16_t elem_bytes;
  uint16_t flags;
  uint64_t meta_addr;
  uint32_t meta_size_bytes;
  uint16_t tmem_base;
  uint16_t meta_tmem_base;
  uint16_t bank_span;
  uint16_t meta_col_span;
  uint8_t tile_role;
  uint8_t payload_kind;
  uint8_t reserved[2];
} tma_descriptor_t;

typedef struct __attribute__((packed)) {
  uint32_t fmt_a;
  uint32_t fmt_b;
  uint32_t fmt_c;
  uint32_t fmt_d;
  uint8_t ws;
  uint8_t sp;
  uint8_t sparse_mode;
  uint8_t transpose_a;
  uint8_t transpose_b;
  uint8_t reserved[3];
  uint16_t a_rows;
  uint16_t a_cols;
  uint16_t b_rows;
  uint16_t b_cols;
  uint16_t c_rows;
  uint16_t c_cols;
} mma_descriptor_t;

#endif
