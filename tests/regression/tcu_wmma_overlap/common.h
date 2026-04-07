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

// Matrix dimensions
#ifndef MATRIX_M
#define MATRIX_M 512
#endif
#ifndef MATRIX_N
#define MATRIX_N 512
#endif
#ifndef MATRIX_K
#define MATRIX_K 512
#endif

// Window shape: 16×16 → A=16×16, B=16×16, C=D=16×16
// 每个 16×16 A/B window 含 2 个 k8 tile → 一次 TMA 搬运 k16
#define WIN_M 16
#define WIN_N 16
#define WIN_K 16

// 硬件计算原语 tile: A=16×8, B=8×16, C/D=16×16
#define TILE_M 16
#define TILE_N 16
#define TILE_K 8

// 每个 window 内的 tile 数
#define A_TILES_PER_WIN ((WIN_M / TILE_M) * (WIN_K / TILE_K))  // 1×2=2
#define B_TILES_PER_WIN ((WIN_K / TILE_K) * (WIN_N / TILE_N))  // 2×1=2

// 全局 tile 计数
#define M_TILES (MATRIX_M / WIN_M)   // 32
#define N_TILES (MATRIX_N / WIN_N)   // 32
#define K_PHASES (MATRIX_K / WIN_K)  // 32

typedef vortex::tensor::descriptor_table_arg_t descriptor_table_arg_t;

typedef struct {
  descriptor_table_arg_t desc_tables;
  uint32_t block_dim[2];
  uint32_t col_span;       // TMEM allocation 列数 (16/32/64)
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
  uint8_t transpose;
  uint8_t reserved;
} tma_descriptor_t;

typedef struct __attribute__((packed)) {
  uint32_t fmt_a;
  uint32_t fmt_b;
  uint32_t fmt_c;
  uint32_t fmt_d;
  uint8_t ws;
  uint8_t sp;
  uint8_t sparse_mode;
  uint8_t reserved[5];
  uint16_t a_rows;
  uint16_t a_cols;
  uint16_t b_rows;
  uint16_t b_cols;
  uint16_t c_rows;
  uint16_t c_cols;
} mma_descriptor_t;

#endif
