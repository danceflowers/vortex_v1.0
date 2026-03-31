#include "common.h"
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context_ab<NUM_THREADS, vt::ATYPE, vt::BTYPE, vt::OTYPE>;

static constexpr uint32_t kTmaADescId = 0;
static constexpr uint32_t kTmaBDescId = 1;
static constexpr uint32_t kTmaCInitDescId = 2;
static constexpr uint32_t kTmaCZeroDescId = 3;
static constexpr uint32_t kTmaOutD0DescId = 4;
static constexpr uint32_t kTmaOutD1DescId = 5;
static constexpr uint32_t kMmaDescId = 0;
static constexpr uint32_t kLoadBarrierId = 0;
static constexpr uint32_t kWmmaBarrierId = 1;
static constexpr uint32_t kStoreDBarrierId = 2;
static constexpr uint32_t kZeroCBarrierId = 3;
static constexpr uint32_t kReloadDBarrierId = 4;
static constexpr uint32_t kStoreReloadedDBarrierId = 5;
static constexpr uint32_t kAWindowId = 0;
static constexpr uint32_t kBWindowId = 1;
static constexpr uint32_t kCWindowId = 2;
static constexpr uint32_t kDWindowId = 3;

static inline void wait_tensor_async(uint32_t barrier_id) {
  (void)vt::tc_commit(barrier_id);
  vt::mbarrier_arrive(barrier_id);
  vt::mbarrier_wait(barrier_id);
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  ctx::fragment_a fragA;
  ctx::fragment_b fragB;
  ctx::fragment_acc fragC;

  ctx::fill_fragment(fragA, 0);
  ctx::fill_fragment(fragB, 0);
  ctx::fill_fragment(fragC, 0);

  uint32_t handle = vt::tmem_alloc(arg->c_bank_span);

  uint32_t a_load_id = vt::tma_load(handle, kTmaADescId, kAWindowId);
  uint32_t b_load_id = vt::tma_load(handle, kTmaBDescId, kBWindowId);
  uint32_t c_init_load_id = vt::tma_load(handle, kTmaCInitDescId, kCWindowId);
  vt::tma_wait(a_load_id);
  vt::tma_wait(b_load_id);
  vt::tma_wait(c_init_load_id);

  vt::mbarrier_init(kLoadBarrierId, 1);
  vt::mma_load_a_slot<kMmaDescId>(handle, kAWindowId, 0, 0);
  vt::mma_load_b_slot<kMmaDescId>(handle, kBWindowId, 0, 0);
  vt::mma_load_c_slot<kMmaDescId>(handle, kCWindowId, 0, 0);
  wait_tensor_async(kLoadBarrierId);

  vt::mbarrier_init(kWmmaBarrierId, 1);
  ctx::mma_sync<kMmaDescId>(fragC, fragA, fragB, fragC);
  wait_tensor_async(kWmmaBarrierId);

  vt::mbarrier_init(kStoreDBarrierId, 1);
  vt::mma_store_c_slot<kMmaDescId>(handle, kDWindowId, 0, 0);
  wait_tensor_async(kStoreDBarrierId);
  uint32_t store_d0_id = vt::tma_store(handle, kTmaOutD0DescId, kDWindowId);
  vt::tma_wait(store_d0_id);

  uint32_t c_zero_load_id = vt::tma_load(handle, kTmaCZeroDescId, kCWindowId);
  vt::tma_wait(c_zero_load_id);

  vt::mbarrier_init(kZeroCBarrierId, 1);
  vt::mma_load_c_slot<kMmaDescId>(handle, kCWindowId, 0, 0);
  wait_tensor_async(kZeroCBarrierId);

  vt::mbarrier_init(kReloadDBarrierId, 1);
  vt::mma_load_c_slot<kMmaDescId>(handle, kDWindowId, 0, 0);
  wait_tensor_async(kReloadDBarrierId);

  vt::mbarrier_init(kStoreReloadedDBarrierId, 1);
  vt::mma_store_c_slot<kMmaDescId>(handle, kDWindowId, 0, 0);
  wait_tensor_async(kStoreReloadedDBarrierId);

  uint32_t store_id = vt::tma_store(handle, kTmaOutD1DescId, kDWindowId);
  vt::tma_wait(store_id);
  vt::tmem_free(handle);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
