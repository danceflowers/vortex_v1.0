#include "common.h"

#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;

namespace {

static constexpr uint32_t kAArgsOff = 0;
static constexpr uint32_t kBArgsOff = 32;
static constexpr uint32_t kCArgsOff = 64;
static constexpr uint32_t kMetaArgsOff = 96;
static constexpr uint32_t kASdescOff = 128;
static constexpr uint32_t kCSdescOff = 136;
static constexpr uint32_t kMetaSdescOff = 144;
static constexpr uint32_t kMbarOff = 160;
static constexpr uint32_t kOpBlockOff = 192;
static constexpr uint32_t kAPayloadOff = 256;
static constexpr uint32_t kBPayloadOff = kAPayloadOff + TCGEN05_PAYLOAD_BYTES;
static constexpr uint32_t kCPayloadOff = kBPayloadOff + TCGEN05_PAYLOAD_BYTES;
static constexpr uint32_t kMetaPayloadOff = kCPayloadOff + TCGEN05_C_PAYLOAD_BYTES;
static constexpr uint32_t kLmemBytes = kMetaPayloadOff + 128;
static constexpr uint32_t kSparseMetaTmemByteOffset = TCGEN05_PAYLOAD_BYTES;

static inline uint64_t make_sdesc(uint32_t lmem_base, uint32_t addr) {
  return static_cast<uint64_t>((addr - lmem_base) / 16);
}

static inline bool is_sparse_case(uint32_t case_id) {
  return case_id == TCGEN05_CASE_SPARSE_2_4_F8
      || case_id == TCGEN05_CASE_SPARSE_1_4_F8;
}

static inline uint32_t c_payload_bytes(uint32_t case_id) {
  return is_sparse_case(case_id) ? TCGEN05_PAYLOAD_BYTES : TCGEN05_C_PAYLOAD_BYTES;
}

static inline void issue_case_mma(uint32_t case_id,
                                  uint32_t d_taddr,
                                  uint32_t a_taddr,
                                  uint64_t b_sdesc,
                                  vt::operand_block_t* op_block) {
  if (case_id == TCGEN05_CASE_ASYM_F16_F8_F32) {
    *op_block = vt::make_operand_block<vt::fp32, vt::fp32>(a_taddr, b_sdesc, 0);
    uint32_t idesc = vt::make_i_descriptor<vt::fp16, vt::fp8, vt::fp32, vt::fp32>(
        M_DIM, N_DIM);
    vt::tcu_mma(d_taddr, idesc, op_block);
    return;
  }

  uint8_t sparsity_kind = (case_id == TCGEN05_CASE_SPARSE_1_4_F8) ? 1 : 0;
  *op_block = vt::make_operand_block<vt::fp16, vt::fp16>(a_taddr, b_sdesc, 0);
  uint32_t idesc = vt::make_i_descriptor<vt::fp8, vt::fp8, vt::fp16, vt::fp16>(
      M_DIM, N_DIM,
      false, false, false, false,
      sparsity_kind,
      0);
  vt::tcu_mma_q<0x5>(d_taddr, idesc, reinterpret_cast<uint32_t>(op_block));
}

} // namespace

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  uint8_t* lmem = reinterpret_cast<uint8_t*>(__local_mem(kLmemBytes));
  uint32_t lmem_base = reinterpret_cast<uint32_t>(__local_mem(0));
  uint32_t a_payload_addr = reinterpret_cast<uint32_t>(lmem + kAPayloadOff);
  uint32_t b_payload_addr = reinterpret_cast<uint32_t>(lmem + kBPayloadOff);
  uint32_t c_payload_addr = reinterpret_cast<uint32_t>(lmem + kCPayloadOff);
  uint32_t meta_payload_addr = reinterpret_cast<uint32_t>(lmem + kMetaPayloadOff);
  uint32_t mbar_addr = reinterpret_cast<uint32_t>(lmem + kMbarOff);

  auto* a_args = reinterpret_cast<vt::cpabulk_transfer_args_t*>(lmem + kAArgsOff);
  auto* b_args = reinterpret_cast<vt::cpabulk_transfer_args_t*>(lmem + kBArgsOff);
  auto* c_args = reinterpret_cast<vt::cpabulk_transfer_args_t*>(lmem + kCArgsOff);
  auto* meta_args = reinterpret_cast<vt::cpabulk_transfer_args_t*>(lmem + kMetaArgsOff);
  *a_args = vt::make_cpabulk_args(a_payload_addr, mbar_addr, 0, 0, 0, 0, 0);
  *b_args = vt::make_cpabulk_args(b_payload_addr, mbar_addr, 0, 0, 0, 0, 0);
  *c_args = vt::make_cpabulk_args(c_payload_addr, mbar_addr, 0, 0, 0, 0, 0);
  *meta_args = vt::make_cpabulk_args(meta_payload_addr, mbar_addr, 0, 0, 0, 0, 0);

  auto* a_tmap = reinterpret_cast<const vt::tensor_map_t*>(
      static_cast<uint32_t>(arg->a_tmap_addr));
  auto* b_tmap = reinterpret_cast<const vt::tensor_map_t*>(
      static_cast<uint32_t>(arg->b_tmap_addr));
  auto* c_tmap = reinterpret_cast<const vt::tensor_map_t*>(
      static_cast<uint32_t>(arg->c_tmap_addr));
  // The mbarrier tx target must match the descriptor payload bytes exactly.
  uint32_t tma_tx_bytes =
      (2 * TCGEN05_PAYLOAD_BYTES) + c_payload_bytes(arg->case_id);
  if (is_sparse_case(arg->case_id)) {
    tma_tx_bytes += TCGEN05_META_BYTES;
  }

  vt::mbarrier_init(mbar_addr, 1);
  vt::mbarrier_expect_tx(mbar_addr, tma_tx_bytes);
  (void)vt::cpabulk_tensor_ld_complete_tx(a_tmap, a_args);
  (void)vt::cpabulk_tensor_ld_complete_tx(b_tmap, b_args);
  (void)vt::cpabulk_tensor_ld_complete_tx(c_tmap, c_args);
  if (is_sparse_case(arg->case_id)) {
    auto* meta_tmap = reinterpret_cast<const vt::tensor_map_t*>(
        static_cast<uint32_t>(arg->meta_tmap_addr));
    (void)vt::cpabulk_tensor_ld_complete_tx(meta_tmap, meta_args);
  }
  uint32_t tma_phase = vt::mbarrier_arrive_token(mbar_addr);
  vt::mbarrier_wait(mbar_addr, tma_phase);

  uint32_t a_taddr = vt::tmem_alloc(16);
  uint32_t d_taddr = vt::tmem_alloc(32);

  uint64_t a_sdesc = make_sdesc(lmem_base, a_payload_addr);
  uint64_t b_sdesc = make_sdesc(lmem_base, b_payload_addr);
  auto* a_sdesc_ptr = reinterpret_cast<uint64_t*>(lmem + kASdescOff);
  *a_sdesc_ptr = a_sdesc;
  vt::tmem_cp_shape<4>(a_taddr, a_sdesc_ptr);

  uint64_t c_sdesc = make_sdesc(lmem_base, c_payload_addr);
  auto* c_sdesc_ptr = reinterpret_cast<uint64_t*>(lmem + kCSdescOff);
  *c_sdesc_ptr = c_sdesc;
  if (arg->case_id == TCGEN05_CASE_ASYM_F16_F8_F32) {
    vt::tmem_cp_shape<3>(d_taddr, c_sdesc_ptr);
  } else {
    vt::tmem_cp_shape<4>(d_taddr, c_sdesc_ptr);
  }

  if (is_sparse_case(arg->case_id)) {
    uint64_t meta_sdesc = make_sdesc(lmem_base, meta_payload_addr);
    auto* meta_sdesc_ptr = reinterpret_cast<uint64_t*>(lmem + kMetaSdescOff);
    *meta_sdesc_ptr = meta_sdesc;
    vt::tmem_cp_shape<1>(a_taddr | (kSparseMetaTmemByteOffset << 16),
                         meta_sdesc_ptr);
  }

  auto* op_block = reinterpret_cast<vt::operand_block_t*>(lmem + kOpBlockOff);

  vt::mbarrier_init(mbar_addr, 1);
  issue_case_mma(arg->case_id, d_taddr, a_taddr, b_sdesc, op_block);
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
