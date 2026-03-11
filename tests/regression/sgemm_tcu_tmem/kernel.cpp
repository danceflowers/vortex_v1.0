#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  auto in_descs = reinterpret_cast<const vt::tma_descriptor_t*>(arg->in_desc_addr);
  auto out_descs = reinterpret_cast<const vt::tma_descriptor_t*>(arg->out_desc_addr);

  ctx::fragment_a fragA;
  ctx::fragment_b fragB;
  ctx::fragment_acc fragC;

  ctx::fill_fragment(fragA, 0);
  ctx::fill_fragment(fragB, 0);
  ctx::fill_fragment(fragC, 0);

  for (uint32_t tile_y = 0; tile_y < arg->tile_grid[1]; ++tile_y) {
    for (uint32_t tile_x = 0; tile_x < arg->tile_grid[0]; ++tile_x) {
      uint32_t tile_id = tile_y * arg->tile_grid[0] + tile_x;
      auto in_desc = &in_descs[tile_id];
      auto out_desc = &out_descs[tile_id];

      uint32_t handle = vt::tmem_alloc(arg->bank_span);
      uint32_t load_id = vt::tma_load(handle, in_desc);
      vt::tma_wait(load_id);

      vt::mma_load<vt::ITYPE, vt::OTYPE>(handle);
      ctx::mma_sync(fragC, fragA, fragB, fragC);
      vt::mma_store<vt::ITYPE, vt::OTYPE>(handle);

      uint32_t store_id = vt::tma_store(handle, out_desc);
      vt::tma_wait(store_id);
      vt::tmem_free(handle);
    }
  }
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  uint32_t grid_dim[2] = {1, 1};
  return vx_spawn_threads(1, grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
