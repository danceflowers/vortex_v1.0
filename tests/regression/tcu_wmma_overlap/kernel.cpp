#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context_ab<NUM_THREADS, vt::ATYPE, vt::BTYPE, vt::OTYPE>;

static constexpr uint32_t kMTiles   = M_TILES;   // 32
static constexpr uint32_t kNTiles   = N_TILES;   // 32
static constexpr uint32_t kKPhases  = K_PHASES;  // 32
static constexpr uint32_t kPipelineStages = 2;
static constexpr uint32_t kMmaDescId = 0;

// ---------------------------------------------------------------------------
// Descriptor-table layout (shared with host):
//   A descs : [0                       .. kMTiles*kKPhases          )
//   B descs : [kBDescBase              .. kBDescBase+kKPhases*kNTiles)
//   C_in    : kCInDescId               (single zero-buffer descriptor)
//   C_out   : [kCOutDescBase           .. kCOutDescBase+kMTiles*kNTiles)
// ---------------------------------------------------------------------------
static constexpr uint32_t kADescCount  = kMTiles * kKPhases;
static constexpr uint32_t kBDescBase   = kADescCount;
static constexpr uint32_t kCInDescId   = kADescCount + kKPhases * kNTiles;
static constexpr uint32_t kCOutDescBase = kCInDescId + 1;

static inline uint32_t a_desc_id(uint32_t m, uint32_t k) {
  return m * kKPhases + k;
}
static inline uint32_t b_desc_id(uint32_t k, uint32_t n) {
  return kBDescBase + k * kNTiles + n;
}
static inline uint32_t c_out_desc_id(uint32_t m, uint32_t n) {
  return kCOutDescBase + m * kNTiles + n;
}

// Barrier IDs
static inline uint32_t ab_barrier_id(uint32_t stage) { return stage; }
static constexpr uint32_t kCLoadBarrier = 2;

// ---------------------------------------------------------------------------
// Async helpers
// ---------------------------------------------------------------------------
static inline void issue_async_mma_load_ab(uint32_t handle_a,
                                           uint32_t handle_b,
                                           uint32_t slot_id,
                                           uint32_t barrier_id) {
  vt::mbarrier_init(barrier_id, 1);
  vt::mma_load_a_slot<kMmaDescId>(handle_a, 0, 0, slot_id);
  vt::mma_load_b_slot<kMmaDescId>(handle_b, 0, 0, slot_id);
  (void)vt::tc_commit(barrier_id);
  vt::mbarrier_arrive(barrier_id);
}

static inline void issue_async_mma_load_c(uint32_t handle_c,
                                          uint32_t slot_id,
                                          uint32_t barrier_id) {
  vt::mbarrier_init(barrier_id, 1);
  vt::mma_load_c_slot<kMmaDescId>(handle_c, 0, 0, slot_id);
  (void)vt::tc_commit(barrier_id);
  vt::mbarrier_arrive(barrier_id);
}

// ---------------------------------------------------------------------------
// kernel_body — each warp processes one row of M-tiles across all N-tiles.
//
// TMEM budget (fp8/fp8/fp16):
//   2 x A-buffers  (ping-pong over K)   = 2 * a_bank_span
//   2 x B-buffers  (ping-pong over K)   = 2 * b_bank_span
//   2 x C-buffers  (ping-pong over N)   = 2 * c_bank_span
//                                       = 2*16 + 2*16 + 2*32 = 128 cols
//
// Pipeline:
//   outer loop : N-tiles  →  C slot ping-pong (slot 0/1)
//   inner loop : K-phases →  A/B slot ping-pong, output-stationary accumulate
// ---------------------------------------------------------------------------
static inline void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t m_tile = blockIdx.x;
  if (m_tile >= kMTiles) return;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragA, 0);
  ctx::fill_fragment(fragB, 0);
  ctx::fill_fragment(fragC, 0);

  // ---- allocate TMEM -------------------------------------------------------
  uint32_t handle_a[kPipelineStages];
  uint32_t handle_b[kPipelineStages];
  uint32_t handle_c[2];
  for (uint32_t s = 0; s < kPipelineStages; ++s) {
    handle_a[s] = vt::tmem_alloc(arg->a_bank_span);
    handle_b[s] = vt::tmem_alloc(arg->b_bank_span);
  }
  handle_c[0] = vt::tmem_alloc(arg->c_bank_span);
  handle_c[1] = vt::tmem_alloc(arg->c_bank_span);

  uint32_t c_store_async[2] = {};
  bool     c_store_live[2]  = {false, false};

  // ---- outer N-tile loop (C ping-pong) ------------------------------------
  for (uint32_t n_tile = 0; n_tile < kNTiles; ++n_tile) {
    uint32_t c_slot = n_tile & 1;

    // wait for the previous tma_store that used this C handle
    if (c_store_live[c_slot]) {
      vt::tma_wait(c_store_async[c_slot]);
      c_store_live[c_slot] = false;
    }

    // load C = 0 : DRAM → TMEM → CMem[c_slot]
    (void)vt::tma_load(handle_c[c_slot], kCInDescId);
    issue_async_mma_load_c(handle_c[c_slot], c_slot, kCLoadBarrier);
    vt::mbarrier_wait(kCLoadBarrier);

    // prefetch first pipeline stages of A/B from DRAM → TMEM
    for (uint32_t s = 0; s < kPipelineStages && s < kKPhases; ++s) {
      (void)vt::tma_load(handle_a[s], a_desc_id(m_tile, s));
      (void)vt::tma_load(handle_b[s], b_desc_id(s, n_tile));
    }
    uint32_t next_prefetch = kPipelineStages;

    // ---- inner K-phase loop (A/B ping-pong, output-stationary) ------------
    for (uint32_t k = 0; k < kKPhases; ++k) {
      uint32_t ab_slot = k & 1;

      // TMEM → AMem/BMem[ab_slot]
      issue_async_mma_load_ab(handle_a[ab_slot], handle_b[ab_slot],
                              ab_slot, ab_barrier_id(ab_slot));
      vt::mbarrier_wait(ab_barrier_id(ab_slot));

      // overlap: prefetch next K-phase into the freed TMEM buffer
      if (next_prefetch < kKPhases) {
        (void)vt::tma_load(handle_a[ab_slot],
                           a_desc_id(m_tile, next_prefetch));
        (void)vt::tma_load(handle_b[ab_slot],
                           b_desc_id(next_prefetch, n_tile));
        ++next_prefetch;
      }

      // WMMA: CMem[c_slot] += AMem[ab_slot] * BMem[ab_slot]
      ctx::mma_sync_slots<kMmaDescId>(ab_slot, ab_slot, c_slot,
                                       fragC, fragA, fragB, fragC);
    }

    // store accumulated output: CMem[c_slot] → TMEM → DRAM
    vt::tc_wait();
    vt::mma_store_c_slot<kMmaDescId>(handle_c[c_slot], 0, 0, c_slot);
    vt::tc_wait();
    c_store_async[c_slot] =
        vt::tma_store(handle_c[c_slot], c_out_desc_id(m_tile, n_tile));
    c_store_live[c_slot] = true;
  }

  // drain any outstanding store
  for (uint32_t s = 0; s < 2; ++s) {
    if (c_store_live[s])
      vt::tma_wait(c_store_async[s]);
  }

  // ---- free TMEM (reverse allocation order) --------------------------------
  vt::tmem_free(handle_c[1]);
  vt::tmem_free(handle_c[0]);
  for (int s = kPipelineStages - 1; s >= 0; --s) {
    vt::tmem_free(handle_b[s]);
    vt::tmem_free(handle_a[s]);
  }
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  uint32_t grid_dim = kMTiles;  // 32 warps = warpgroup
  return vx_spawn_threads(1, &grid_dim, arg->block_dim,
                          (vx_kernel_func_cb)kernel_body, arg);
}
