#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;

static constexpr uint32_t kWarpgroupWarps  = 4;
static constexpr uint32_t kLocalBarrierId  = 0;
static constexpr uint32_t kTmaADescId      = 0;
static constexpr uint32_t kTmaBDescId      = 1;
static constexpr uint32_t kTmaCDescId      = 2;
static constexpr uint32_t kTmaOutDescId    = 3;

struct shared_state_t {
  uint32_t handle;
};

static inline void warpgroup_sync() {
  vx_barrier(kLocalBarrierId, kWarpgroupWarps);
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  auto shared = reinterpret_cast<shared_state_t*>(__local_mem(sizeof(shared_state_t)));

  if (vx_warp_id() == 0) {
    shared->handle = vt::tmem_alloc(arg->c_bank_span);
  }

  warpgroup_sync();

  uint32_t handle = shared->handle;

  // Phase-2: cpabulk routes through legacy TmaFrontend (GAP-1).
  (void)vt::cpabulk_tensor_ld(kTmaADescId, /*window=*/0);
  (void)vt::cpabulk_tensor_ld(kTmaBDescId, /*window=*/1);
  (void)vt::cpabulk_tensor_ld(kTmaCDescId, /*window=*/2);
  vt::tcu_wait_ld();

  // Single TCU_MMA replaces fragment-based mma_load/sync/store sequence.
  vt::operand_block_t op = vt::make_operand_block(handle, /*b_sdesc=*/0);
  uint32_t idesc = vt::make_idescriptor<vt::ATYPE, vt::BTYPE, vt::OTYPE, vt::OTYPE>(0, 0);
  vt::tcu_mma(/*d_taddr=*/handle, idesc, &op);
  vt::tcu_wait_st();

  (void)vt::cpabulk_tensor_st(kTmaOutDescId, /*window=*/3);
  vt::tcu_wait_st();

  warpgroup_sync();

  if (vx_warp_id() == 0) {
    vt::tmem_dealloc(handle, arg->c_bank_span);
  }

  warpgroup_sync();
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
