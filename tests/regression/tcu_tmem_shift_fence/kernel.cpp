#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;

static constexpr uint32_t kTmaInDescId = 0;
static constexpr uint32_t kTmaOutDescId = 1;

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t handle = vt::tmem_alloc(arg->bank_span);

  // cp.async.bulk.tensor → TMEM (Phase-2 cut routes through legacy TmaFrontend
  // with rs1=desc_id; args_lmem_ptr currently ignored, see GAP-1).
  uint32_t load_id = vt::cpabulk_tensor_ld(kTmaInDescId, /*coords_ctl=*/0);
  (void)load_id;
  vt::tcu_wait_ld();

  vt::mbar_fence_before();

  uint32_t shift_id = vt::tmem_shift(handle);
  (void)shift_id;
  vt::tcu_wait_ld();

  vt::mbar_fence_after();

  uint32_t store_id = vt::cpabulk_tensor_st(kTmaOutDescId, /*coords_ctl=*/0);
  (void)store_id;
  vt::tcu_wait_st();

  vt::tmem_dealloc(handle, arg->bank_span);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
