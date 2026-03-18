#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context_ab<NUM_THREADS, vt::ATYPE, vt::BTYPE, vt::OTYPE>;

static constexpr uint32_t kTileCount = 4;
static constexpr uint32_t kADescBase = 0;
static constexpr uint32_t kBDescBase = kTileCount;
static constexpr uint32_t kCInDescBase = 2 * kTileCount;
static constexpr uint32_t kCOutDescBase = 3 * kTileCount;
static constexpr uint32_t kMmaDescId = 0;
static constexpr uint32_t kMaxWorkerCount = 2;
static constexpr uint32_t kPipelineStages = 2;
static constexpr uint32_t kBarrierStride = kPipelineStages * 3;
static constexpr uint32_t kTmemBanks = 16;

static inline uint32_t load_barrier_id(uint32_t worker_id, uint32_t stage) {
  return worker_id * kBarrierStride + stage;
}

static inline uint32_t wmma_barrier_id(uint32_t worker_id, uint32_t stage) {
  return worker_id * kBarrierStride + kPipelineStages + stage;
}

static inline uint32_t store_barrier_id(uint32_t worker_id, uint32_t stage) {
  return worker_id * kBarrierStride + 2 * kPipelineStages + stage;
}

static inline void issue_async_mma_load(uint32_t handle_a,
                                        uint32_t handle_b,
                                        uint32_t handle_c,
                                        uint32_t barrier_id) {
  vt::mbarrier_init(barrier_id, 1);
  vt::mma_load_a<kMmaDescId>(handle_a);
  vt::mma_load_b<kMmaDescId>(handle_b);
  vt::mma_load_c<kMmaDescId>(handle_c);
  (void)vt::tc_commit(barrier_id);
  vt::mbarrier_arrive(barrier_id);
}

static inline void issue_async_wmma(ctx::fragment_acc& fragC,
                                    const ctx::fragment_a& fragA,
                                    const ctx::fragment_b& fragB,
                                    uint32_t barrier_id) {
  vt::mbarrier_init(barrier_id, 1);
  ctx::mma_sync<kMmaDescId>(fragC, fragA, fragB, fragC);
  (void)vt::tc_commit(barrier_id);
  vt::mbarrier_arrive(barrier_id);
}

static inline void issue_async_mma_store(uint32_t handle_c, uint32_t barrier_id) {
  vt::mbarrier_init(barrier_id, 1);
  vt::mma_store_c<kMmaDescId>(handle_c);
  (void)vt::tc_commit(barrier_id);
  vt::mbarrier_arrive(barrier_id);
}

static inline void run_worker(kernel_arg_t* __UNIFORM__ arg, uint32_t worker_id) {
  ctx::fragment_a fragA;
  ctx::fragment_b fragB;
  ctx::fragment_acc fragC;

  ctx::fill_fragment(fragA, 0);
  ctx::fill_fragment(fragB, 0);
  ctx::fill_fragment(fragC, 0);

  uint32_t per_worker_span = 2 * arg->a_bank_span + 2 * arg->b_bank_span + arg->c_bank_span;
  uint32_t worker_count = (per_worker_span * kMaxWorkerCount <= kTmemBanks) ? kMaxWorkerCount : 1;
  if (worker_id >= worker_count) {
    return;
  }

  uint32_t handle_a[kPipelineStages];
  uint32_t handle_b[kPipelineStages];
  for (uint32_t stage = 0; stage < kPipelineStages; ++stage) {
    handle_a[stage] = vt::tmem_alloc(arg->a_bank_span);
    handle_b[stage] = vt::tmem_alloc(arg->b_bank_span);
  }
  uint32_t handle_c = vt::tmem_alloc(arg->c_bank_span);

  uint32_t a_load_id[kPipelineStages] = {};
  uint32_t b_load_id[kPipelineStages] = {};
  uint32_t next_tile = worker_id;
  if (next_tile >= kTileCount) {
    return;
  }

  uint32_t current_stage = 0;
  a_load_id[current_stage] = vt::tma_load(handle_a[current_stage], kADescBase + next_tile);
  b_load_id[current_stage] = vt::tma_load(handle_b[current_stage], kBDescBase + next_tile);
  next_tile += worker_count;

  for (uint32_t tile_id = worker_id; tile_id < kTileCount; tile_id += worker_count) {
    uint32_t stage = current_stage;
    uint32_t next_stage = (current_stage + 1) % kPipelineStages;

    vt::tma_wait(a_load_id[stage]);
    vt::tma_wait(b_load_id[stage]);

    uint32_t c_load_id = vt::tma_load(handle_c, kCInDescBase + tile_id);
    vt::tma_wait(c_load_id);
    issue_async_mma_load(handle_a[stage], handle_b[stage], handle_c, load_barrier_id(worker_id, stage));

    if (next_tile < kTileCount) {
      a_load_id[next_stage] = vt::tma_load(handle_a[next_stage], kADescBase + next_tile);
      b_load_id[next_stage] = vt::tma_load(handle_b[next_stage], kBDescBase + next_tile);
    }

    vt::mbarrier_wait(load_barrier_id(worker_id, stage));
    issue_async_wmma(fragC, fragA, fragB, wmma_barrier_id(worker_id, stage));

    vt::mbarrier_wait(wmma_barrier_id(worker_id, stage));
    issue_async_mma_store(handle_c, store_barrier_id(worker_id, stage));

    vt::mbarrier_wait(store_barrier_id(worker_id, stage));
    uint32_t store_id = vt::tma_store(handle_c, kCOutDescBase + tile_id);
    vt::tma_wait(store_id);

    if (next_tile < kTileCount) {
      next_tile += worker_count;
    }

    current_stage = next_stage;
  }

  vt::tmem_free(handle_c);
  for (uint32_t stage = 0; stage < kPipelineStages; ++stage) {
    vt::tmem_free(handle_b[stage]);
    vt::tmem_free(handle_a[stage]);
  }
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t worker_id = blockIdx.x;
  if (worker_id >= kMaxWorkerCount) {
    return;
  }
  run_worker(arg, worker_id);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  uint32_t worker_warps = kMaxWorkerCount;
  return vx_spawn_threads(1, &worker_warps, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
