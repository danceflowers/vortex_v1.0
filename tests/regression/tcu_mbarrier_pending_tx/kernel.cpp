#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;

// Phase-3.2 end-to-end validation for GAP-1 (DRAM->LMEM cpabulk data path)
// and GAP-2 (mbarrier::complete_tx::bytes wiring).
//
// Sequence:
//   1. Carve LMEM space for: mbarrier_state_t, cpabulk_transfer_args_t, dest
//   2. mbarrier_init(N=1)            (LMEM mbar object)
//   3. mbarrier_expect_tx(N_bytes)   (set expected_tx_count)
//   4. cpabulk_tensor_ld typed overload — Core takes the LMEM-direct path:
//        - reads tensor_map_t (128 B) from DRAM via host-prepared address
//        - reads cpabulk_transfer_args_t (32 B) from LMEM
//        - copies N bytes DRAM -> LMEM
//        - on completion fires mbarrier_complete_tx(mbar_addr, N)
//   5. mbarrier_wait blocks until phase parity flips (drives by GAP-2 hook)
//   6. Read first u32 of LMEM dest and write to DRAM out_buffer for the host
//      to verify (proves the bytes physically landed in LMEM = GAP-1)

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  // Allocate one LMEM region big enough for: mbar (8B) + args (32B) + N bytes.
  uint8_t* lmem = reinterpret_cast<uint8_t*>(__local_mem(8 + 32 + 4096));

  uint32_t mbar_addr     = reinterpret_cast<uint32_t>(lmem + 0);
  uint32_t args_addr     = reinterpret_cast<uint32_t>(lmem + 8);
  uint32_t dest_smem_off = reinterpret_cast<uint32_t>(lmem + 8 + 32);

  // Build the cpabulk_transfer_args_t in LMEM. coords[]=0 fetches tile 0.
  vt::cpabulk_transfer_args_t* args =
      reinterpret_cast<vt::cpabulk_transfer_args_t*>(args_addr);
  *args = vt::make_cpabulk_args(/*smem_addr=*/dest_smem_off,
                                /*mbar_addr=*/mbar_addr,
                                /*coords[0..4]=*/0, 0, 0, 0, 0);

  vt::mbarrier_init(mbar_addr, /*count=*/1);
  vt::mbarrier_expect_tx(mbar_addr, arg->tx_bytes);

  // qualifier[5]=mbarrier::complete_tx::bytes variant. Core sees args_lmem_ptr
  // in LMEM range, does DRAM->LMEM copy, then on completion fires
  // mbarrier_complete_tx(args.mbar_addr, bytes_transferred). GAP-2.
  uint32_t tmap_addr32 = static_cast<uint32_t>(arg->tensor_map_addr);
  (void)vt::cpabulk_tensor_ld_complete_tx(tmap_addr32, args_addr);

  // Single thread acknowledges its arrival (init count was 1). After this
  // pending_arrival_count==0 and the only thing left for phase advance is
  // pending_tx_count >= expected_tx_count, which the cpabulk completion
  // hook drives.
  vt::mbarrier_arrive(mbar_addr);

  // Block until phase parity flips. Unblocks once both arrival and tx
  // pending counts have reached their targets (PTX §7.6.2).
  vt::mbarrier_wait(mbar_addr, /*phase_token=*/0);

  // Hand back the first u32 of the destination so the host can confirm the
  // copy actually landed in LMEM and the wait did unblock.
  uint32_t first_word = *reinterpret_cast<volatile uint32_t*>(dest_smem_off);
  *reinterpret_cast<volatile uint32_t*>(arg->out_addr) = first_word;
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(1, arg->grid_dim, arg->block_dim,
                          (vx_kernel_func_cb)kernel_body, arg);
}
