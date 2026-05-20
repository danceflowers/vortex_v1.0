#include "common.h"

#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  uint8_t* lmem = reinterpret_cast<uint8_t*>(
      __local_mem(8 + sizeof(vt::cpabulk_transfer_args_t) + TMA_STORE_BYTES));
  uint32_t mbar_addr = reinterpret_cast<uint32_t>(lmem);
  uint32_t args_addr = reinterpret_cast<uint32_t>(lmem + 8);
  uint32_t payload_addr = reinterpret_cast<uint32_t>(
      lmem + 8 + sizeof(vt::cpabulk_transfer_args_t));

  volatile uint32_t* payload = reinterpret_cast<volatile uint32_t*>(payload_addr);
  for (uint32_t i = 0; i < TMA_STORE_WORDS; ++i) {
    payload[i] = 0x5A5A0000u | (i & 0xffffu);
  }

  auto* args = reinterpret_cast<vt::cpabulk_transfer_args_t*>(args_addr);
  *args = vt::make_cpabulk_args(payload_addr, 0, 0, 0, 0, 0, 0);
  auto* tmap = reinterpret_cast<const vt::tensor_map_t*>(
      static_cast<uint32_t>(arg->tensor_map_addr));

  vt::mbarrier_init(mbar_addr, 1);
  (void)vt::cpabulk_tensor_st(tmap, args);
  (void)vt::mbar_commit(mbar_addr);
  uint32_t phase = vt::mbarrier_arrive_token(mbar_addr);
  vt::mbarrier_wait(mbar_addr, phase);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(1, arg->grid_dim, arg->block_dim,
                          (vx_kernel_func_cb)kernel_body, arg);
}
