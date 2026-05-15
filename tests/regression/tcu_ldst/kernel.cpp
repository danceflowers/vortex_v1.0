#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;

// Phase-3.4 Stage 0 smoke: tcgen05.{ld,st} with PTX TADDR encoding
// (PTX §9.7.16.1):
//   taddr[15:0]  = lane base
//   taddr[31:16] = col_byte offset (1-byte cell — Vortex deviation)
//
// tmem_alloc returns a TADDR directly: lane=col_base, col_byte=0.
// All threads issue the same warp-uniform taddr; per-thread t the simx side
// computes actual_lane = taddr.lane + t and accesses 4 cells starting at
// (actual_lane, taddr.col_byte). The 32-thread warp therefore writes to lanes
// [col_base .. col_base+31], so col_span must >= NUM_THREADS=32.

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t base_taddr = vt::tmem_alloc(/*col_span=*/32);

  uint32_t lane = vx_thread_id();
  uint32_t pattern = 0xCAFE0000u | (lane & 0xFFFFu);

  vt::tcu_st(base_taddr, pattern);
  vt::tcu_wait_st();

  uint32_t observed = vt::tcu_ld(base_taddr);
  vt::tcu_wait_ld();

  uint32_t* out = reinterpret_cast<uint32_t*>(arg->out_addr);
  out[lane] = observed;

  vt::tmem_dealloc(base_taddr, 32);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(1, arg->grid_dim, arg->block_dim,
                          (vx_kernel_func_cb)kernel_body, arg);
}
