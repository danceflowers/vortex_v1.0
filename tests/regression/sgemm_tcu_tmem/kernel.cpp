#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;

// Phase-2 migration. The original sgemm_tcu_tmem orchestrated a sophisticated
// pipeline using legacy fragment-based mma_sync_slots, slot-based mma_load/
// store_*_slot fan-out, and tc_commit_scope::mma_load_only scope hints. None
// of these exist in the new ISA: TCU_MMA fans out fill→compute→drain in a
// single instruction, and mbar_commit registers ALL inflight tcgen05.async
// ops (no scope restriction).
//
// This skeleton preserves the test's macro structure (per-tile, per-phase
// double-buffered K-loop) using only Phase-2 constructs. Data-path
// correctness depends on GAP-1/2/3/4 (DRAM↔LMEM, complete_tx, tmem_cp,
// tcu_ld/st) closing.

static constexpr uint32_t kPipelineStages = 2;
static constexpr uint32_t kWorkerWarps    = WORKER_WARPS;

static inline uint32_t a_desc_base(const kernel_arg_t* /*arg*/) { return 0; }
static inline uint32_t b_desc_base(const kernel_arg_t* arg) {
  uint32_t num_tiles = arg->tile_grid[0] * arg->tile_grid[1];
  return arg->phase_limit * num_tiles;
}
static inline uint32_t c_in_desc_base(const kernel_arg_t* arg) {
  uint32_t num_tiles = arg->tile_grid[0] * arg->tile_grid[1];
  return 2 * arg->phase_limit * num_tiles;
}
static inline uint32_t c_out_desc_base(const kernel_arg_t* arg) {
  uint32_t num_tiles = arg->tile_grid[0] * arg->tile_grid[1];
  return 2 * arg->phase_limit * num_tiles + num_tiles;
}
static inline uint32_t phase_desc_id(uint32_t base, uint32_t phase,
                                     uint32_t num_tiles, uint32_t tile_id) {
  return base + phase * num_tiles + tile_id;
}

static inline void free_worker_handles(uint32_t handle_a[kPipelineStages],
                                       uint32_t handle_b[kPipelineStages],
                                       uint32_t handle_c,
                                       uint32_t handle_d,
                                       uint32_t col_span) {
  vt::tmem_dealloc(handle_d, col_span);
  vt::tmem_dealloc(handle_c, col_span);
  for (uint32_t stage = 0; stage < kPipelineStages; ++stage) {
    vt::tmem_dealloc(handle_b[stage], col_span);
    vt::tmem_dealloc(handle_a[stage], col_span);
  }
}

static inline void run_tile(kernel_arg_t* __UNIFORM__ arg,
                            uint32_t tile_id,
                            uint32_t handle_a[kPipelineStages],
                            uint32_t handle_b[kPipelineStages],
                            uint32_t handle_c,
                            uint32_t handle_d,
                            uint32_t idesc) {
  uint32_t num_tiles = arg->tile_grid[0] * arg->tile_grid[1];
  (void)num_tiles;

  // Stage C-init.
  (void)vt::cpabulk_tensor_ld(c_in_desc_base(arg) + tile_id, /*window=*/0);

  // Stage A/B for phase 0 / 1.
  if (arg->phase_limit > 0) {
    (void)vt::cpabulk_tensor_ld(phase_desc_id(a_desc_base(arg), 0, num_tiles, tile_id), 0);
    (void)vt::cpabulk_tensor_ld(phase_desc_id(b_desc_base(arg), 0, num_tiles, tile_id), 1);
  }
  if (arg->phase_limit > 1) {
    (void)vt::cpabulk_tensor_ld(phase_desc_id(a_desc_base(arg), 1, num_tiles, tile_id), 0);
    (void)vt::cpabulk_tensor_ld(phase_desc_id(b_desc_base(arg), 1, num_tiles, tile_id), 1);
  }
  vt::tcu_wait_ld();

  for (uint32_t phase_base = 0; phase_base < arg->phase_limit;
       phase_base += kPipelineStages) {
    uint32_t cur = phase_base & 1;

    if (phase_base < arg->phase_limit) {
      vt::operand_block_t op = vt::make_operand_block(handle_a[cur], /*b_sdesc=*/0);
      vt::tcu_mma(/*d_taddr=*/handle_d, idesc, &op);
    }
    if (phase_base + 1 < arg->phase_limit) {
      vt::operand_block_t op = vt::make_operand_block(handle_a[1 - cur], /*b_sdesc=*/0);
      vt::tcu_mma(/*d_taddr=*/handle_d, idesc, &op);
    }
    if (phase_base + 2 < arg->phase_limit) {
      (void)vt::cpabulk_tensor_ld(phase_desc_id(a_desc_base(arg), phase_base + 2, num_tiles, tile_id), 0);
      (void)vt::cpabulk_tensor_ld(phase_desc_id(b_desc_base(arg), phase_base + 2, num_tiles, tile_id), 1);
    }
    if (phase_base + 3 < arg->phase_limit) {
      (void)vt::cpabulk_tensor_ld(phase_desc_id(a_desc_base(arg), phase_base + 3, num_tiles, tile_id), 0);
      (void)vt::cpabulk_tensor_ld(phase_desc_id(b_desc_base(arg), phase_base + 3, num_tiles, tile_id), 1);
    }
  }

  vt::tcu_wait_st();
  (void)vt::cpabulk_tensor_st(c_out_desc_base(arg) + tile_id, /*window=*/3);
  (void)handle_b;
  (void)handle_c;
}

static inline void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  if (vx_warp_id() >= kWorkerWarps) {
    return;
  }

  uint32_t handle_a[kPipelineStages];
  uint32_t handle_b[kPipelineStages];
  for (uint32_t stage = 0; stage < kPipelineStages; ++stage) {
    handle_a[stage] = vt::tmem_alloc(arg->a_bank_span);
    handle_b[stage] = vt::tmem_alloc(arg->b_bank_span);
  }
  uint32_t handle_c = vt::tmem_alloc(arg->c_bank_span);
  uint32_t handle_d = vt::tmem_alloc(arg->c_bank_span);

  uint32_t idesc = vt::make_idescriptor<vt::ATYPE, vt::BTYPE, vt::OTYPE, vt::OTYPE>(
      /*M=*/0, /*N=*/0);

  uint32_t num_tiles = arg->tile_grid[0] * arg->tile_grid[1];
  for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
    run_tile(arg, tile_id, handle_a, handle_b, handle_c, handle_d, idesc);
  }

  vt::tcu_wait_st();
  free_worker_handles(handle_a, handle_b, handle_c, handle_d, arg->c_bank_span);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  uint32_t worker_warps = kWorkerWarps;
  return vx_spawn_threads(1, &worker_warps, arg->block_dim,
                          (vx_kernel_func_cb)kernel_body, arg);
}
