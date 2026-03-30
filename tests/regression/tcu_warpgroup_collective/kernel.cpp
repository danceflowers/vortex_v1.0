#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context_ab<NUM_THREADS, vt::ATYPE, vt::BTYPE, vt::OTYPE>;

static constexpr uint32_t kWarpgroupWarps = 4;
static constexpr uint32_t kLocalBarrierId = 0;
static constexpr uint32_t kTmaADescId = 0;
static constexpr uint32_t kTmaBDescId = 1;
static constexpr uint32_t kTmaCDescId = 2;
static constexpr uint32_t kTmaOutDescId = 3;
static constexpr uint32_t kMmaDescId = 0;
static constexpr uint32_t kSlotId = 0;

struct shared_state_t {
  uint32_t handle;
};

static inline void warpgroup_sync() {
  vx_barrier(kLocalBarrierId, kWarpgroupWarps);
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = reinterpret_cast<shared_state_t*>(__local_mem(sizeof(shared_state_t)));

  ctx::fragment_a fragA;
  ctx::fragment_b fragB;
  ctx::fragment_acc fragC;

  ctx::fill_fragment(fragA, 0);
  ctx::fill_fragment(fragB, 0);
  ctx::fill_fragment(fragC, 0);

  if (vx_warp_id() == 0) {
    shared->handle = vt::tmem_alloc(arg->c_bank_span);
  }

  warpgroup_sync();

  uint32_t handle = shared->handle;

  uint32_t load_a_id = vt::tma_load(handle, kTmaADescId, 0);
  uint32_t load_b_id = vt::tma_load(handle, kTmaBDescId, 1);
  uint32_t load_c_id = vt::tma_load(handle, kTmaCDescId, 2);
  vt::tma_wait(load_a_id);
  vt::tma_wait(load_b_id);
  vt::tma_wait(load_c_id);

  vt::mma_load_a_slot<kMmaDescId>(handle, 0, 0, kSlotId);
  vt::mma_load_b_slot<kMmaDescId>(handle, 1, 0, kSlotId);
  vt::mma_load_c_slot<kMmaDescId>(handle, 2, 0, kSlotId);
  vt::tc_wait();

  ctx::mma_sync_slots<kMmaDescId>(kSlotId, kSlotId, kSlotId, fragC, fragA, fragB, fragC);
  vt::tc_wait();

  vt::mma_store_c_slot<kMmaDescId>(handle, 3, 0, kSlotId);
  vt::tc_wait();

  uint32_t store_id = vt::tma_store(handle, kTmaOutDescId, 3);
  vt::tma_wait(store_id);

  warpgroup_sync();

  if (vx_warp_id() == 0) {
    vt::tmem_free(handle);
  }

  warpgroup_sync();
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
