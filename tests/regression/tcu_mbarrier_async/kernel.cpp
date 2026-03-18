#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context_ab<NUM_THREADS, vt::ATYPE, vt::BTYPE, vt::OTYPE>;

static constexpr uint32_t kTmaInDescId = 0;
static constexpr uint32_t kTmaOutDescId = 1;
static constexpr uint32_t kMmaDescId = 0;
static constexpr uint32_t kMmaLoadBarrierId = 2;
static constexpr uint32_t kWmmaBarrierId = 3;
static constexpr uint32_t kMmaStoreBarrierId = 4;

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

  uint32_t handle = vt::tmem_alloc(arg->bank_span);

  vt::mbarrier_init(0, 1);
  (void)vt::tma_load(handle, kTmaInDescId);
  (void)vt::tc_commit(0);
  vt::tc_fence_before();
  vt::mbarrier_arrive(0);
  vt::mbarrier_wait(0);
  vt::tc_fence_after();

  vt::mbarrier_init(kMmaLoadBarrierId, 1);
  vt::mma_load<kMmaDescId>(handle);
  wait_tensor_async(kMmaLoadBarrierId);
  vt::mbarrier_init(kWmmaBarrierId, 1);
  ctx::mma_sync<kMmaDescId>(fragC, fragA, fragB, fragC);
  wait_tensor_async(kWmmaBarrierId);
  vt::mbarrier_init(kMmaStoreBarrierId, 1);
  vt::mma_store<kMmaDescId>(handle);
  wait_tensor_async(kMmaStoreBarrierId);

  vt::mbarrier_init(1, 1);
  (void)vt::tma_store(handle, kTmaOutDescId);
  (void)vt::tc_commit(1);
  vt::tc_fence_before();
  vt::mbarrier_arrive(1);
  vt::mbarrier_wait(1);
  vt::tc_fence_after();

  vt::tmem_free(handle);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
