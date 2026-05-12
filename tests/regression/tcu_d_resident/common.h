#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <tensor_cfg.h>

#ifndef NUM_THREADS
#define NUM_THREADS 32
#endif

// 16x16x16 fp16-mma microkernel exercised K_ITERS times back-to-back with
// enable_input_d=1; each iteration accumulates A[k]*B[k] into the resident
// D collector (DMem in fp22) without writing TMEM until the final mma.

#define M_DIM    16
#define N_DIM    16
#define K_DIM    16
#define K_ITERS  4

typedef struct {
  uint32_t grid_dim[2];
  uint32_t block_dim[2];
  uint64_t a_addr;       // K_ITERS * M*K fp16 source
  uint64_t b_addr;       // K_ITERS * K*N fp16 source
  uint64_t d_addr;       // M*N fp32 result
} kernel_arg_t;

#endif
