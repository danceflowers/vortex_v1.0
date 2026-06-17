#include "common.h"

#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;

namespace {

static constexpr uint32_t kAArgsOff = 0;
static constexpr uint32_t kBArgsOff = 32;
static constexpr uint32_t kAPoisonArgsOff = 64;
static constexpr uint32_t kASdescOff = 96;
static constexpr uint32_t kAPoisonSdescOff = 104;
static constexpr uint32_t kMbarOff = 112;
static constexpr uint32_t kOpBlockOff = 128;
static constexpr uint32_t kAPayloadOff = 192;
static constexpr uint32_t kBPayloadOff = kAPayloadOff + TCGEN05_PAYLOAD_BYTES;
static constexpr uint32_t kAPoisonPayloadOff = kBPayloadOff + TCGEN05_PAYLOAD_BYTES;
static constexpr uint32_t kLmemBytes = kAPoisonPayloadOff + TCGEN05_PAYLOAD_BYTES;

static inline uint64_t make_sdesc(uint32_t lmem_base, uint32_t addr) {
  return static_cast<uint64_t>((addr - lmem_base) / 16);
}

static inline void run_mma_and_wait(uint32_t mbar_addr,
                                    uint32_t d_taddr,
                                    uint32_t idesc,
                                    vt::operand_block_t* op_block,
                                    uint32_t qualifier) {
  vt::mbarrier_init(mbar_addr, 1);
  if (qualifier == 0x00) {
    vt::tcu_mma_q<0x00>(d_taddr, idesc, reinterpret_cast<uint32_t>(op_block));
  } else if (qualifier == 0x10) {
    vt::tcu_mma_q<0x10>(d_taddr, idesc, reinterpret_cast<uint32_t>(op_block));
  } else {
    vt::tcu_mma_q<0x20>(d_taddr, idesc, reinterpret_cast<uint32_t>(op_block));
  }
  (void)vt::mbar_commit(mbar_addr);
  uint32_t phase = vt::mbarrier_arrive_token(mbar_addr);
  vt::mbarrier_wait(mbar_addr, phase);
  vt::mbar_fence_after();
}

static inline void read_d_to_output(uint32_t d_taddr,
                                    volatile uint32_t* out_words,
                                    uint32_t result_idx) {
  uint32_t lane = vx_thread_id();
  uint32_t base = result_idx * NUM_TCU_LD_WORDS * 32;
  uint32_t d0 = vt::tcu_ld(d_taddr | ((0u * sizeof(uint32_t)) << 16));
  uint32_t d1 = vt::tcu_ld(d_taddr | ((1u * sizeof(uint32_t)) << 16));
  uint32_t d2 = vt::tcu_ld(d_taddr | ((2u * sizeof(uint32_t)) << 16));
  uint32_t d3 = vt::tcu_ld(d_taddr | ((3u * sizeof(uint32_t)) << 16));
  uint32_t d4 = vt::tcu_ld(d_taddr | ((4u * sizeof(uint32_t)) << 16));
  uint32_t d5 = vt::tcu_ld(d_taddr | ((5u * sizeof(uint32_t)) << 16));
  uint32_t d6 = vt::tcu_ld(d_taddr | ((6u * sizeof(uint32_t)) << 16));
  uint32_t d7 = vt::tcu_ld(d_taddr | ((7u * sizeof(uint32_t)) << 16));
  vt::tcu_wait_ld();
  out_words[base + 0 * 32 + lane] = d0;
  out_words[base + 1 * 32 + lane] = d1;
  out_words[base + 2 * 32 + lane] = d2;
  out_words[base + 3 * 32 + lane] = d3;
  out_words[base + 4 * 32 + lane] = d4;
  out_words[base + 5 * 32 + lane] = d5;
  out_words[base + 6 * 32 + lane] = d6;
  out_words[base + 7 * 32 + lane] = d7;
}

} // namespace

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  uint8_t* lmem = reinterpret_cast<uint8_t*>(__local_mem(kLmemBytes));
  uint32_t lmem_base = reinterpret_cast<uint32_t>(__local_mem(0));
  uint32_t a_payload_addr = reinterpret_cast<uint32_t>(lmem + kAPayloadOff);
  uint32_t b_payload_addr = reinterpret_cast<uint32_t>(lmem + kBPayloadOff);
  uint32_t a_poison_payload_addr = reinterpret_cast<uint32_t>(lmem + kAPoisonPayloadOff);
  uint32_t mbar_addr = reinterpret_cast<uint32_t>(lmem + kMbarOff);

  auto* a_args = reinterpret_cast<vt::cpabulk_transfer_args_t*>(lmem + kAArgsOff);
  auto* b_args = reinterpret_cast<vt::cpabulk_transfer_args_t*>(lmem + kBArgsOff);
  auto* a_poison_args =
      reinterpret_cast<vt::cpabulk_transfer_args_t*>(lmem + kAPoisonArgsOff);
  *a_args = vt::make_cpabulk_args(a_payload_addr, mbar_addr, 0, 0, 0, 0, 0);
  *b_args = vt::make_cpabulk_args(b_payload_addr, mbar_addr, 0, 0, 0, 0, 0);
  *a_poison_args = vt::make_cpabulk_args(a_poison_payload_addr, mbar_addr, 0, 0, 0, 0, 0);

  auto* a_tmap = reinterpret_cast<const vt::tensor_map_t*>(
      static_cast<uint32_t>(arg->a_tmap_addr));
  auto* b_tmap = reinterpret_cast<const vt::tensor_map_t*>(
      static_cast<uint32_t>(arg->b_tmap_addr));
  auto* a_poison_tmap = reinterpret_cast<const vt::tensor_map_t*>(
      static_cast<uint32_t>(arg->a_poison_tmap_addr));

  vt::mbarrier_init(mbar_addr, 1);
  vt::mbarrier_expect_tx(mbar_addr, 3 * TCGEN05_PAYLOAD_BYTES);
  (void)vt::cpabulk_tensor_ld_complete_tx(a_tmap, a_args);
  (void)vt::cpabulk_tensor_ld_complete_tx(b_tmap, b_args);
  (void)vt::cpabulk_tensor_ld_complete_tx(a_poison_tmap, a_poison_args);
  uint32_t tma_phase = vt::mbarrier_arrive_token(mbar_addr);
  vt::mbarrier_wait(mbar_addr, tma_phase);

  uint32_t a_taddr = vt::tmem_alloc(16);
  uint32_t d_taddr = vt::tmem_alloc(32);

  uint64_t a_sdesc = make_sdesc(lmem_base, a_payload_addr);
  uint64_t poison_sdesc = make_sdesc(lmem_base, a_poison_payload_addr);
  uint64_t b_sdesc = make_sdesc(lmem_base, b_payload_addr);
  auto* a_sdesc_ptr = reinterpret_cast<uint64_t*>(lmem + kASdescOff);
  auto* poison_sdesc_ptr = reinterpret_cast<uint64_t*>(lmem + kAPoisonSdescOff);
  *a_sdesc_ptr = a_sdesc;
  *poison_sdesc_ptr = poison_sdesc;

  vt::tmem_cp_shape<4>(a_taddr, a_sdesc_ptr);

  auto* op_block = reinterpret_cast<vt::operand_block_t*>(lmem + kOpBlockOff);
  *op_block = vt::make_operand_block<vt::fp32, vt::fp32>(a_taddr, b_sdesc, 0);
  uint32_t idesc = vt::make_i_descriptor<vt::fp16, vt::fp16, vt::fp32, vt::fp32>(
      M_DIM, N_DIM);

  auto* out_words = reinterpret_cast<volatile uint32_t*>(
      static_cast<uint32_t>(arg->out_addr));

  run_mma_and_wait(mbar_addr, d_taddr, idesc, op_block, 0x00);

  // If USE/LASTUSE incorrectly refetch A from TMEM, they will see all zeros
  // instead of the identity tile captured by the collector FILL.
  vt::tmem_cp_shape<4>(a_taddr, poison_sdesc_ptr);

  run_mma_and_wait(mbar_addr, d_taddr, idesc, op_block, 0x10);
  run_mma_and_wait(mbar_addr, d_taddr, idesc, op_block, 0x20);
  read_d_to_output(d_taddr, out_words, 0);

  vt::tmem_dealloc(d_taddr, 32);
  vt::tmem_dealloc(a_taddr, 16);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(1, arg->grid_dim, arg->block_dim,
                          (vx_kernel_func_cb)kernel_body, arg);
}
