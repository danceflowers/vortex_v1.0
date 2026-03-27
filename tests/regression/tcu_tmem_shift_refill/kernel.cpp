#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;

static constexpr uint32_t kWindowId = 1;
static constexpr uint32_t kTmaInDescId = 0;
static constexpr uint32_t kTmaRefillDescId = 1;
static constexpr uint32_t kTmaOutDescId = 2;

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t handle = vt::tmem_alloc(arg->bank_span);

  uint32_t load_id = vt::tma_load(handle, kTmaInDescId, kWindowId);
  vt::tma_wait(load_id);

  vt::tc_fence_before();

  uint32_t shift_id = vt::tmem_shift_refill(handle, kWindowId, kTmaRefillDescId);
  vt::tma_wait(shift_id);

  vt::tc_fence_after();

  uint32_t store_id = vt::tma_store(handle, kTmaOutDescId, kWindowId);
  vt::tma_wait(store_id);

  vt::tmem_free(handle);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
