#include "common.h"

#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  constexpr uint32_t kAArgsOff = 0;
  constexpr uint32_t kBArgsOff = 32;
  constexpr uint32_t kMbarOff = 72;
  constexpr uint32_t kOpBlockOff = 80;
  constexpr uint32_t kAPayloadOff = 128;
  constexpr uint32_t kBPayloadOff = kAPayloadOff + TCGEN05_PAYLOAD_BYTES;
  constexpr uint32_t kLmemBytes = kBPayloadOff + TCGEN05_PAYLOAD_BYTES;

  uint8_t* lmem = reinterpret_cast<uint8_t*>(__local_mem(kLmemBytes));
  uint32_t lmem_base = reinterpret_cast<uint32_t>(__local_mem(0));
  uint32_t b_payload_addr = reinterpret_cast<uint32_t>(lmem + kBPayloadOff);
  uint32_t mbar_addr = reinterpret_cast<uint32_t>(lmem + kMbarOff);

  // Allocate TMEM up front so the TMA can deliver A straight into TMEM (A,D live
  // in TMEM; B lives in LMEM).
  uint32_t a_taddr = vt::tmem_alloc(16);
  uint32_t d_taddr = vt::tmem_alloc(32);

  auto* a_args = reinterpret_cast<vt::cpabulk_transfer_args_t*>(lmem + kAArgsOff);
  auto* b_args = reinterpret_cast<vt::cpabulk_transfer_args_t*>(lmem + kBArgsOff);
  // A → TMEM (smem_addr field carries the TMEM taddr); B → LMEM (b_payload).
  *a_args = vt::make_cpabulk_args(a_taddr, mbar_addr, 0, 0, 0, 0, 0);
  *b_args = vt::make_cpabulk_args(b_payload_addr, mbar_addr, 0, 0, 0, 0, 0);

  auto* a_tmap = reinterpret_cast<const vt::tensor_map_t*>(
      static_cast<uint32_t>(arg->a_tmap_addr));
  auto* b_tmap = reinterpret_cast<const vt::tensor_map_t*>(
      static_cast<uint32_t>(arg->b_tmap_addr));
  vt::mbarrier_init(mbar_addr, 1);
  vt::mbarrier_expect_tx(mbar_addr, 2 * TCGEN05_PAYLOAD_BYTES);
  (void)vt::cpabulk_tensor_ld_complete_tx(a_tmap, a_args);
  (void)vt::cpabulk_tensor_ld_complete_tx(b_tmap, b_args);
  uint32_t tma_phase = vt::mbarrier_arrive_token(mbar_addr);
  vt::mbarrier_wait(mbar_addr, tma_phase);

  uint64_t b_sdesc = static_cast<uint64_t>((b_payload_addr - lmem_base) / 16);

  auto* op_block = reinterpret_cast<vt::operand_block_t*>(lmem + kOpBlockOff);
  *op_block = vt::make_operand_block(a_taddr, b_sdesc, 0);

  uint32_t idesc = vt::make_i_descriptor<vt::fp16, vt::fp16, vt::fp32, vt::fp32>(
      M_DIM, N_DIM);
  vt::mbarrier_init(mbar_addr, 1);
  vt::tcu_mma_no_accum(d_taddr, idesc, op_block);
  (void)vt::mbar_commit(mbar_addr);
  uint32_t phase = vt::mbarrier_arrive_token(mbar_addr);
  vt::mbarrier_wait(mbar_addr, phase);
  vt::mbar_fence_after();

  uint32_t lane = vx_thread_id();
  auto* out_words = reinterpret_cast<volatile uint32_t*>(
      static_cast<uint32_t>(arg->out_addr));
  uint32_t d0 = vt::tcu_ld(d_taddr | ((0u * sizeof(uint32_t)) << 16));
  uint32_t d1 = vt::tcu_ld(d_taddr | ((1u * sizeof(uint32_t)) << 16));
  uint32_t d2 = vt::tcu_ld(d_taddr | ((2u * sizeof(uint32_t)) << 16));
  uint32_t d3 = vt::tcu_ld(d_taddr | ((3u * sizeof(uint32_t)) << 16));
  uint32_t d4 = vt::tcu_ld(d_taddr | ((4u * sizeof(uint32_t)) << 16));
  uint32_t d5 = vt::tcu_ld(d_taddr | ((5u * sizeof(uint32_t)) << 16));
  uint32_t d6 = vt::tcu_ld(d_taddr | ((6u * sizeof(uint32_t)) << 16));
  uint32_t d7 = vt::tcu_ld(d_taddr | ((7u * sizeof(uint32_t)) << 16));
  vt::tcu_wait_ld();
  out_words[0 * 32 + lane] = d0;
  out_words[1 * 32 + lane] = d1;
  out_words[2 * 32 + lane] = d2;
  out_words[3 * 32 + lane] = d3;
  out_words[4 * 32 + lane] = d4;
  out_words[5 * 32 + lane] = d5;
  out_words[6 * 32 + lane] = d6;
  out_words[7 * 32 + lane] = d7;

  vt::tmem_dealloc(d_taddr, 32);
  vt::tmem_dealloc(a_taddr, 16);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(1, arg->grid_dim, arg->block_dim,
                          (vx_kernel_func_cb)kernel_body, arg);
}
