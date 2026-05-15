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

  // Phase-2 cpabulk routes through legacy TmaFrontend (GAP-1).
  uint32_t load_id = vt::cpabulk_tensor_ld(kTmaInDescId, /*coords_ctl=*/kWindowId);
  (void)load_id;
  vt::tcu_wait_ld();

  vt::mbar_fence_before();

  uint32_t shift_id = vt::tmem_shift_refill(handle, kWindowId, kTmaRefillDescId);
  (void)shift_id;
  vt::tcu_wait_ld();

  vt::mbar_fence_after();

  uint32_t store_id = vt::cpabulk_tensor_st(kTmaOutDescId, /*coords_ctl=*/kWindowId);
  (void)store_id;
  vt::tcu_wait_st();

  vt::tmem_dealloc(handle, arg->bank_span);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
