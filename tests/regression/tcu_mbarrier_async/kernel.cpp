#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;

// Phase-2: legacy fragment-based mma_sync + mma_load/store_*_slot replaced by
// a single TCU_MMA (custom-3 funct3=000). MBAR_INIT/ARRIVE/WAIT/COMMIT
// reorganised under custom-2 (0x5B). cpabulk_tensor_ld currently bridges
// through legacy TmaFrontend (GAP-1).

static constexpr uint32_t kTmaInDescId = 0;
static constexpr uint32_t kTmaBDescId  = 1;
static constexpr uint32_t kTmaCDescId  = 2;
static constexpr uint32_t kTmaOutDescId = 3;

// One mbarrier slot reused across the producer / wmma / store steps.
static constexpr uint32_t kBarLoad  = 0;
static constexpr uint32_t kBarWmma  = 0;
static constexpr uint32_t kBarStore = 0;

static inline void wait_tensor_async(uint32_t mbar) {
  (void)vt::mbar_commit(mbar);
  vt::mbarrier_arrive(mbar);
  vt::mbarrier_wait(mbar);
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t handle = vt::tmem_alloc(arg->c_bank_span);

  // Producer: stage A/B/C in TMEM via cpabulk.
  vt::mbarrier_init(0, 1);
  (void)vt::cpabulk_tensor_ld(kTmaInDescId, /*window=*/0);
  (void)vt::cpabulk_tensor_ld(kTmaBDescId,  /*window=*/1);
  (void)vt::cpabulk_tensor_ld(kTmaCDescId,  /*window=*/2);
  (void)vt::mbar_commit(0);
  vt::mbar_fence_before();
  vt::mbarrier_arrive(0);
  vt::mbarrier_wait(0);
  vt::mbar_fence_after();

  // Compute: TCU_MMA fans out fill→compute→drain internally.
  vt::mbarrier_init(kBarLoad, 1);
  vt::operand_block_t op = vt::make_operand_block(handle, /*b_sdesc=*/0);
  uint32_t idesc = vt::make_idescriptor<vt::ATYPE, vt::BTYPE, vt::OTYPE, vt::OTYPE>(0, 0);
  vt::tcu_mma(/*d_taddr=*/handle, idesc, &op);
  wait_tensor_async(kBarWmma);
  vt::mbarrier_init(kBarStore, 1);
  // No separate drain instruction in Phase-2: TCU_MMA already drains DMem→TMEM.
  wait_tensor_async(kBarStore);

  // Drain to DRAM.
  vt::mbarrier_init(1, 1);
  (void)vt::cpabulk_tensor_st(kTmaOutDescId, /*window=*/3);
  (void)vt::mbar_commit(1);
  vt::mbar_fence_before();
  vt::mbarrier_arrive(1);
  vt::mbarrier_wait(1);
  vt::mbar_fence_after();

  vt::tmem_dealloc(handle, arg->c_bank_span);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
