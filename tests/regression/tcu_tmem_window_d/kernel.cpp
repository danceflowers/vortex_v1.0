#include "common.h"
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;

// Phase-2 migration: legacy fragment-based mma_sync + mma_load/store_*_slot
// macros are gone. The new ISA expresses the entire fill→compute→drain chain
// in a single TCU_MMA instruction. Multi-window orchestration (D-window load
// vs C-window load on the same handle) becomes a sequence of TCU_MMA + cpabulk
// calls; the real D-window addressing depends on tcgen05.cp data path
// (GAP-3) for shared→TMEM staging.

static constexpr uint32_t kTmaADescId         = 0;
static constexpr uint32_t kTmaBDescId         = 1;
static constexpr uint32_t kTmaCInitDescId     = 2;
static constexpr uint32_t kTmaCZeroDescId     = 3;
static constexpr uint32_t kTmaOutD0DescId     = 4;
static constexpr uint32_t kTmaOutD1DescId     = 5;

static constexpr uint32_t kBarLoad         = 0;
static constexpr uint32_t kBarWmma         = 1;
static constexpr uint32_t kBarStoreD       = 2;
static constexpr uint32_t kBarZeroC        = 3;
static constexpr uint32_t kBarReloadD      = 4;
static constexpr uint32_t kBarStoreReloadD = 5;

static constexpr uint32_t kAWindowId = 0;
static constexpr uint32_t kBWindowId = 1;
static constexpr uint32_t kCWindowId = 2;
static constexpr uint32_t kDWindowId = 3;

static inline void wait_tensor_async(uint32_t mbar) {
  (void)vt::mbar_commit(mbar);
  vt::mbarrier_arrive(mbar);
  vt::mbarrier_wait(mbar);
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t handle = vt::tmem_alloc(arg->c_bank_span);

  (void)vt::cpabulk_tensor_ld(kTmaADescId,     kAWindowId);
  (void)vt::cpabulk_tensor_ld(kTmaBDescId,     kBWindowId);
  (void)vt::cpabulk_tensor_ld(kTmaCInitDescId, kCWindowId);
  vt::tcu_wait_ld();

  vt::mbarrier_init(kBarLoad, 1);
  wait_tensor_async(kBarLoad);

  vt::operand_block_t op = vt::make_operand_block(handle, /*b_sdesc=*/0);
  uint32_t idesc = vt::make_idescriptor<vt::ATYPE, vt::BTYPE, vt::OTYPE, vt::OTYPE>(0, 0);
  vt::mbarrier_init(kBarWmma, 1);
  vt::tcu_mma(/*d_taddr=*/handle, idesc, &op);
  wait_tensor_async(kBarWmma);

  vt::mbarrier_init(kBarStoreD, 1);
  // No separate drain instruction: TCU_MMA already produced D in TMEM.
  wait_tensor_async(kBarStoreD);
  (void)vt::cpabulk_tensor_st(kTmaOutD0DescId, kDWindowId);
  vt::tcu_wait_st();

  // Reset C and re-accumulate from D (R-M-W).
  (void)vt::cpabulk_tensor_ld(kTmaCZeroDescId, kCWindowId);
  vt::tcu_wait_ld();

  vt::mbarrier_init(kBarZeroC, 1);
  wait_tensor_async(kBarZeroC);

  vt::mbarrier_init(kBarReloadD, 1);
  wait_tensor_async(kBarReloadD);

  vt::mbarrier_init(kBarStoreReloadD, 1);
  // PTX d_taddr R-M-W: a second TCU_MMA on the same d_taddr accumulates.
  vt::tcu_mma(/*d_taddr=*/handle, idesc, &op);
  vt::tcu_wait_st();
  wait_tensor_async(kBarStoreReloadD);

  (void)vt::cpabulk_tensor_st(kTmaOutD1DescId, kDWindowId);
  vt::tcu_wait_st();
  vt::tmem_dealloc(handle, arg->c_bank_span);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
