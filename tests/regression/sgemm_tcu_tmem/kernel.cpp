#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context_ab<NUM_THREADS, vt::ATYPE, vt::BTYPE, vt::OTYPE>;

static constexpr uint32_t kMmaDescId = 0;
static constexpr uint32_t kTensorBarrierStride = 3;
static constexpr uint32_t kTensorLoadBarrier = 0;
static constexpr uint32_t kTensorComputeBarrier = 1;
static constexpr uint32_t kTensorStoreBarrier = 2;

static inline uint32_t tensor_barrier_id(uint32_t worker_id, uint32_t stage) {
  return worker_id * kTensorBarrierStride + stage;
}

static inline void wait_tensor_async(uint32_t barrier_id) {
  (void)vt::tc_commit(barrier_id);
  vt::mbarrier_arrive(barrier_id);
  vt::mbarrier_wait(barrier_id);
}

static inline void run_worker(kernel_arg_t* __UNIFORM__ arg,
                              uint32_t worker_id,
                              uint32_t worker_count) {
  ctx::fragment_a fragA;
  ctx::fragment_b fragB;
  ctx::fragment_acc fragC;

  ctx::fill_fragment(fragA, 0);
  ctx::fill_fragment(fragB, 0);
  ctx::fill_fragment(fragC, 0);

  uint32_t handle0 = vt::tmem_alloc(arg->bank_span);

  uint32_t num_tiles = arg->tile_grid[0] * arg->tile_grid[1];
  for (uint32_t tile_id = worker_id; tile_id < num_tiles; tile_id += worker_count) {
    uint32_t load_id = vt::tma_load(handle0, tile_id);
    vt::tma_wait(load_id);

    vt::mbarrier_init(tensor_barrier_id(worker_id, kTensorLoadBarrier), 1);
    vt::mma_load<kMmaDescId>(handle0);
    wait_tensor_async(tensor_barrier_id(worker_id, kTensorLoadBarrier));
    vt::mbarrier_init(tensor_barrier_id(worker_id, kTensorComputeBarrier), 1);
    ctx::mma_sync<kMmaDescId>(fragC, fragA, fragB, fragC);
    wait_tensor_async(tensor_barrier_id(worker_id, kTensorComputeBarrier));
    vt::mbarrier_init(tensor_barrier_id(worker_id, kTensorStoreBarrier), 1);
    vt::mma_store<kMmaDescId>(handle0);
    wait_tensor_async(tensor_barrier_id(worker_id, kTensorStoreBarrier));

    uint32_t store_id = vt::tma_store(handle0, num_tiles + tile_id);
    vt::tma_wait(store_id);
  }

  vt::tmem_free(handle0);
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t num_tiles = arg->tile_grid[0] * arg->tile_grid[1];
  if (0 == num_tiles) {
    return;
  }

  uint32_t worker_id = blockIdx.x;
  uint32_t worker_count = gridDim.x;
  if (worker_id >= worker_count) {
    return;
  }
  run_worker(arg, worker_id, worker_count);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  uint32_t num_tiles = arg->tile_grid[0] * arg->tile_grid[1];
  constexpr uint32_t kSimxTmemBanks = 16;
  uint32_t worker_warps = kSimxTmemBanks / arg->bank_span;
  if (worker_warps == 0) {
    worker_warps = 1;
  }
  auto available_warps = static_cast<uint32_t>(vx_num_warps());
  if (worker_warps > available_warps) {
    worker_warps = available_warps;
  }
  uint32_t worker_cap = 2;
  if (worker_warps > worker_cap) {
    worker_warps = worker_cap;
  }
  if (worker_warps > num_tiles) {
    worker_warps = num_tiles;
  }
  return vx_spawn_threads(1, &worker_warps, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
