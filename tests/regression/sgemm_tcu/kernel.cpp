#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;

// Phase-2 migration. sgemm_tcu was a fragment-based WMMA test; the underlying
// instruction (RISCV_CUSTOM0 funct7=2) is gone after the custom-1/2/3 ISA
// realignment. The Phase-2 GEMM compute path is:
//   1. tmem_alloc
//   2. cpabulk_tensor_ld: stage A/B/C from DRAM to TMEM
//   3. tcu_mma: TCU computes D = A·B + D in-place at d_taddr
//   4. cpabulk_tensor_st: drain D to DRAM
//
// This skeleton exercises the new ISA encoding paths. Full data-path
// correctness requires the Phase-3.2 GAPs to close (GAP-1 DRAM↔LMEM,
// GAP-3 tmem_cp shape×decompress, plus the host-side runtime to populate
// tensor_map_t descriptors instead of raw A_addr/B_addr/C_addr).

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;

  uint32_t handle = vt::tmem_alloc(/*col_span=*/64);

  uint32_t idesc = vt::make_idescriptor<vt::ITYPE, vt::ITYPE, vt::OTYPE, vt::OTYPE>(
      static_cast<uint16_t>(M), static_cast<uint16_t>(N));

  // Naive K-loop: each iteration accumulates a tcu_mma into d_taddr (R-M-W).
  for (uint32_t i = 0; i < K; ++i) {
    // Stage A and B tiles into TMEM. Phase-2 cpabulk routes through legacy
    // TmaFrontend (GAP-1); rs1 is reinterpreted as descriptor index, so this
    // path needs the runtime side to provide tensor_map_t entries to be
    // genuinely correct.
    (void)vt::cpabulk_tensor_ld(/*tensor_map=*/0, /*coords_ctl=*/i);
    (void)vt::cpabulk_tensor_ld(/*tensor_map=*/1, /*coords_ctl=*/i);
    vt::tcu_wait_ld();

    vt::operand_block_t op = vt::make_operand_block(handle, /*b_sdesc=*/0);
    vt::tcu_mma(/*d_taddr=*/handle, idesc, &op);
  }
  vt::tcu_wait_st();

  (void)vt::cpabulk_tensor_st(/*tensor_map=*/2, /*coords_ctl=*/0);
  vt::tcu_wait_st();

  vt::tmem_dealloc(handle, /*ncols=*/64);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
