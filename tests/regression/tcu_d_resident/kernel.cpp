#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;

// Phase-3.3.2 D output-resident regression (GAP-11).
//
// STATUS: scaffold + compile clean; runtime hangs because the legacy
// enqueue_async_mma_load path (called by tcu_mma's internal fan-out) requires
// a TmemWindowPlan binding that tcu_st-seeded TMEM doesn't establish. See
// docs/tcgen05_alignment_gap.txt Phase-3.3.9 for the fill-byte-range refactor
// that unblocks this regression.
//
// Goal: validate that K consecutive TCU_MMA calls with enable_input_d=1
// accumulate correctly, where the running accumulator stays resident in
// DMem (fp22) across iterations and only the final store_d writes back to
// TMEM at d_taddr.
//
// Flow:
//   1. tmem_alloc handle_a / handle_b / handle_d (single shared col_span)
//   2. Per-thread: pull A[k] / B[k] u32 words from DRAM, write into TMEM via
//      tcu_st (4 u32 per lane × K_ITERS, since 16*16*fp16 = 512 B = 4 u32 / lane)
//   3. Build operand_block in LMEM (a_taddr=handle_a, b_sdesc_lo=handle_b)
//   4. First mma: tcu_mma_no_accum (qualifier[0]=0) → D = A[0]·B[0]
//   5. Iters 1..K-1: tcu_mma (qualifier[0]=1) → D += A[k]·B[k]
//   6. tcu_wait_st
//   7. Per-thread: tcu_ld 8 u32 words from handle_d → write to DRAM result
//
// Host compares observed against fp32 CPU reference within ULP tolerance.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t handle_a = vt::tmem_alloc(/*col_span=*/16);
  uint32_t handle_b = vt::tmem_alloc(/*col_span=*/16);
  uint32_t handle_d = vt::tmem_alloc(/*col_span=*/16);

  uint32_t lane = vx_thread_id();

  // Per-lane source pointers (each lane owns 4 u32 words of A/B per iter).
  // A is 16*16 fp16 = 256 fp16 = 128 u32 words total per iter; with 32 lanes
  // each lane handles 4 u32 words. Same for B.
  // Actually 16*16 fp16 = 256 elements × 2 bytes = 512 bytes = 128 u32 words.
  // 128 / 32 = 4 u32 words per lane.

  // ---- Stage 1: per-iter K_ITERS, seed A[k] and B[k] into TMEM via tcu_st ----
  // We collapse A/B fills inside the K loop along with mma calls.

  // operand_block_t in LMEM (per-warp scratch).
  uint8_t* lmem = reinterpret_cast<uint8_t*>(__local_mem(64));
  vt::operand_block_t* op_block =
      reinterpret_cast<vt::operand_block_t*>(lmem);
  *op_block = vt::make_operand_block(
      /*a_taddr=*/handle_a,
      /*b_sdesc=*/static_cast<uint64_t>(handle_b),
      /*lanes_off=*/0);

  uint32_t idesc = vt::make_idescriptor<vt::fp16, vt::fp16, vt::fp32, vt::fp32>(
      /*M=*/16, /*N=*/16);

  uint32_t a_taddr = handle_a;       // base_off=0 in low byte form
  uint32_t b_taddr = handle_b;
  uint32_t d_taddr = handle_d;

  for (uint32_t k = 0; k < K_ITERS; ++k) {
    // Per-iter A/B loads from DRAM into RF, then TCU_ST into TMEM.
    // Each lane writes 4 u32 words at consecutive byte offsets (lane*4 within
    // each "chunk" of 32 lanes; 4 chunks of 128 bytes each = 512 B = 4 chunks).
    auto a_src = reinterpret_cast<volatile const uint32_t*>(arg->a_addr) + k * 128;
    auto b_src = reinterpret_cast<volatile const uint32_t*>(arg->b_addr) + k * 128;
    for (uint32_t chunk = 0; chunk < 4; ++chunk) {
      uint32_t a_word = a_src[chunk * 32 + lane];
      uint32_t b_word = b_src[chunk * 32 + lane];
      uint32_t off_in_taddr = (chunk * 128) << 8;  // base_off in taddr high bytes
      vt::tcu_st(a_taddr | off_in_taddr, a_word);
      vt::tcu_st(b_taddr | off_in_taddr, b_word);
    }
    vt::tcu_wait_st();

    // Issue mma. First iter: no_accum (D = A·B). Subsequent: accumulate.
    if (k == 0) {
      vt::tcu_mma_no_accum(d_taddr, idesc, op_block);
    } else {
      vt::tcu_mma(d_taddr, idesc, op_block);
    }
    vt::tcu_wait_st();
  }

  // ---- Drain: read D from TMEM (1024 B = 256 u32 = 8 u32 / lane) ----
  auto d_dst = reinterpret_cast<volatile uint32_t*>(arg->d_addr);
  for (uint32_t chunk = 0; chunk < 8; ++chunk) {
    uint32_t off_in_taddr = (chunk * 128) << 8;
    uint32_t w = vt::tcu_ld(d_taddr | off_in_taddr);
    d_dst[chunk * 32 + lane] = w;
  }
  vt::tcu_wait_ld();

  vt::tmem_dealloc(handle_d, 16);
  vt::tmem_dealloc(handle_b, 16);
  vt::tmem_dealloc(handle_a, 16);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(1, arg->grid_dim, arg->block_dim,
                          (vx_kernel_func_cb)kernel_body, arg);
}
