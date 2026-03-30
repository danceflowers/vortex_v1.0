#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context_ab<NUM_THREADS, vt::ATYPE, vt::BTYPE, vt::OTYPE>;

static constexpr bool kSparse2To4 = (SPARSE_MODE == 1);
static constexpr uint32_t kPhaseCount = 4;
static constexpr uint32_t kWorkerWarps = WORKER_WARPS;
static constexpr uint32_t kTileCount = TILE_COUNT;
static constexpr uint32_t kMmaDescId = 0;
static constexpr uint32_t kPipelineStages = 2;
static constexpr uint32_t kCSlot = 0;
static constexpr uint32_t kCLoadBarrier = 2;
static constexpr uint32_t kDualWarpLoadBarrierStride = 2;

static_assert(kWorkerWarps >= 1 && kWorkerWarps <= 2, "tcu_wmma_overlap supports 1 or 2 worker warps");
static_assert(kTileCount >= kWorkerWarps && kTileCount <= 2, "tcu_wmma_overlap supports up to 2 tiles");

static inline uint32_t a_desc_id(uint32_t tile_id, uint32_t phase) {
  return tile_id * kPhaseCount + phase;
}

static inline uint32_t b_desc_base() {
  return kTileCount * kPhaseCount;
}

static inline uint32_t b_desc_id(uint32_t tile_id, uint32_t phase) {
  return b_desc_base() + tile_id * kPhaseCount + phase;
}

static inline uint32_t c_in_desc_base() {
  return 2 * kTileCount * kPhaseCount;
}

static inline uint32_t c_in_desc_id(uint32_t tile_id) {
  return c_in_desc_base() + tile_id;
}

static inline uint32_t c_out_desc_base() {
  return c_in_desc_base() + kTileCount;
}

static inline uint32_t c_out_desc_id(uint32_t tile_id) {
  return c_out_desc_base() + tile_id;
}

static inline uint32_t load_barrier_id(uint32_t stage) {
  return stage;
}

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

static inline void free_handles(uint32_t handle_a[kPipelineStages],
                                uint32_t handle_b[kPipelineStages],
                                uint32_t handle_c,
                                uint32_t handle_d) {
  vt::tmem_free(handle_d);
  vt::tmem_free(handle_c);
  for (uint32_t stage = 0; stage < kPipelineStages; ++stage) {
    vt::tmem_free(handle_b[stage]);
    vt::tmem_free(handle_a[stage]);
  }
}

static inline void run_worker_single(kernel_arg_t* __UNIFORM__ arg, uint32_t tile_id) {
  ctx::fragment_a fragA;
  ctx::fragment_b fragB;
  ctx::fragment_acc fragC;

  ctx::fill_fragment(fragA, 0);
  ctx::fill_fragment(fragB, 0);
  ctx::fill_fragment(fragC, 0);

  uint32_t handle_a[kPipelineStages];
  uint32_t handle_b[kPipelineStages];
  for (uint32_t stage = 0; stage < kPipelineStages; ++stage) {
    handle_a[stage] = vt::tmem_alloc(arg->a_bank_span);
    handle_b[stage] = vt::tmem_alloc(arg->b_bank_span);
  }
  uint32_t handle_c = vt::tmem_alloc(arg->c_bank_span);
  uint32_t handle_d = vt::tmem_alloc(arg->c_bank_span);

  uint32_t a_load_id[kPipelineStages] = {};
  uint32_t b_load_id[kPipelineStages] = {};
  for (uint32_t stage = 0; stage < kPipelineStages && stage < kPhaseCount; ++stage) {
    a_load_id[stage] = vt::tma_load(handle_a[stage], a_desc_id(tile_id, stage));
    b_load_id[stage] = vt::tma_load(handle_b[stage], b_desc_id(tile_id, stage));
  }
  (void)vt::tma_load(handle_c, c_in_desc_id(tile_id));
  uint32_t next_phase_to_prefetch = kPipelineStages;

  issue_async_mma_load_c(handle_c, kCSlot, kCLoadBarrier);
  vt::mbarrier_wait(kCLoadBarrier);

  for (uint32_t phase = 0; phase < kPhaseCount; ++phase) {
    uint32_t ab_slot = phase & 0x1;

    issue_async_mma_load_ab(handle_a[ab_slot], handle_b[ab_slot], ab_slot, load_barrier_id(ab_slot));
    vt::mbarrier_wait(load_barrier_id(ab_slot));

    if (next_phase_to_prefetch < kPhaseCount) {
      a_load_id[ab_slot] = vt::tma_load(handle_a[ab_slot], a_desc_id(tile_id, next_phase_to_prefetch));
      b_load_id[ab_slot] = vt::tma_load(handle_b[ab_slot], b_desc_id(tile_id, next_phase_to_prefetch));
      ++next_phase_to_prefetch;
    }

    ctx::mma_sync_slots<kMmaDescId>(ab_slot, ab_slot, kCSlot, fragC, fragA, fragB, fragC);
  }

  vt::tc_wait();
  vt::mma_store_c_slot<kMmaDescId>(handle_d, 0, 0, kCSlot);
  vt::tc_wait();
  auto store_id = vt::tma_store(handle_d, c_out_desc_id(tile_id));
  vt::tma_wait(store_id);

  free_handles(handle_a, handle_b, handle_c, handle_d);
}

static inline void run_worker_dual(kernel_arg_t* __UNIFORM__ arg, uint32_t tile_id, uint32_t slot_id) {
  ctx::fragment_a fragA;
  ctx::fragment_b fragB;
  ctx::fragment_acc fragC;

  ctx::fill_fragment(fragA, 0);
  ctx::fill_fragment(fragB, 0);
  ctx::fill_fragment(fragC, 0);

  uint32_t handle_a = vt::tmem_alloc(arg->a_bank_span);
  uint32_t handle_b = vt::tmem_alloc(arg->b_bank_span);
  uint32_t handle_c = vt::tmem_alloc(arg->c_bank_span);
  uint32_t handle_d = vt::tmem_alloc(arg->c_bank_span);

  uint32_t ab_load_barrier = slot_id * kDualWarpLoadBarrierStride + 0;
  uint32_t c_load_barrier = slot_id * kDualWarpLoadBarrierStride + 1;

  (void)vt::tma_load(handle_c, c_in_desc_id(tile_id));
  issue_async_mma_load_c(handle_c, slot_id, c_load_barrier);
  vt::mbarrier_wait(c_load_barrier);

  for (uint32_t phase = 0; phase < kPhaseCount; ++phase) {
    (void)vt::tma_load(handle_a, a_desc_id(tile_id, phase));
    (void)vt::tma_load(handle_b, b_desc_id(tile_id, phase));
    issue_async_mma_load_ab(handle_a, handle_b, slot_id, ab_load_barrier);
    vt::mbarrier_wait(ab_load_barrier);
    ctx::mma_sync_slots<kMmaDescId>(slot_id, slot_id, slot_id, fragC, fragA, fragB, fragC);
  }

  vt::tc_wait();
  vt::mma_store_c_slot<kMmaDescId>(handle_d, 0, 0, slot_id);
  vt::tc_wait();
  auto store_id = vt::tma_store(handle_d, c_out_desc_id(tile_id));
  vt::tma_wait(store_id);

  vt::tmem_free(handle_d);
  vt::tmem_free(handle_c);
  vt::tmem_free(handle_b);
  vt::tmem_free(handle_a);
}

static inline void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  if (blockIdx.x >= kWorkerWarps) {
    return;
  }
  if constexpr (kWorkerWarps == 1) {
    run_worker_single(arg, 0);
  } else {
    run_worker_dual(arg, blockIdx.x, vx_warp_id() & 0x1);
  }
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  uint32_t worker_warps = kWorkerWarps;
  return vx_spawn_threads(1, &worker_warps, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
