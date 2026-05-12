#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;

// Phase-2 migration: legacy fragment-based mma_sync + mma_load/store_*_slot
// macros are gone. The new ISA expresses the entire fill→compute→drain chain
// in a single TCU_MMA instruction (custom-3 funct3=000). cpabulk_tensor_ld/st
// currently route through the legacy TmaFrontend (GAP-1); rs2=args_lmem_ptr
// is accepted but ignored until DRAM→LMEM path lands.

static constexpr uint32_t kTmaInDescId = 0;
static constexpr uint32_t kTmaBDescId  = 1;
static constexpr uint32_t kTmaCDescId  = 2;
static constexpr uint32_t kTmaOutDescId = 3;

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t a_handle = vt::tmem_alloc(arg->c_bank_span);

  // Stage A/B/C tiles into TMEM via cpabulk.
  (void)vt::cpabulk_tensor_ld(kTmaInDescId, /*window=*/0);
  (void)vt::cpabulk_tensor_ld(kTmaBDescId,  /*window=*/1);
  (void)vt::cpabulk_tensor_ld(kTmaCDescId,  /*window=*/2);
  vt::tcu_wait_ld();

  // tcgen05.mma — d_taddr is read-modify-written (PTX d_taddr R-M-W).
  // operand_block_t lives in LMEM; we declare it in a thread-local scope and
  // pass its address as rs2.
  vt::operand_block_t op = vt::make_operand_block(a_handle, /*b_sdesc=*/0);
  uint32_t idesc = vt::make_idescriptor<vt::ATYPE, vt::BTYPE, vt::OTYPE, vt::OTYPE>(
      /*M=*/0, /*N=*/0);
  vt::tcu_mma(/*d_taddr=*/a_handle, idesc, &op);
  vt::tcu_wait_st();

  // Drain D back out to DRAM via cpabulk.tensor.st.
  (void)vt::cpabulk_tensor_st(kTmaOutDescId, /*window=*/3);
  vt::tcu_wait_st();
  vt::tmem_dealloc(a_handle, arg->c_bank_span);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
