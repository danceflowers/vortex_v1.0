// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <tensor_cfg.h>
#include <vx_intrinsics.h>

namespace vortex {
namespace tensor {

enum mem_layout {
  row_major,
  col_major
};

static constexpr uint32_t tmem_alloc_reserved_operand = 0xffffffffu;

// ============================================================================
// PTX-aligned descriptor types — mirror sim/simx/tensor/idescriptor.h.
// Kernel-side declarations so user code can construct on-LMEM blobs that simx
// (and future RTL) can read byte-for-byte.
// ============================================================================

// Bit-packed 32-bit tcgen05.mma instruction descriptor (PTX §9.7.16.5.4).
struct i_descriptor_t {
  uint32_t sparsity_meta_sel : 2;  // [1:0]
  uint32_t sparsity_kind     : 1;  // [2]
  uint32_t reserved_3        : 1;  // [3]
  uint32_t kind              : 4;  // [7:4]
  uint32_t shape_m           : 8;  // [15:8]
  uint32_t shape_n           : 8;  // [23:16]
  uint32_t a_storage_layout  : 2;  // [25:24]
  uint32_t b_storage_layout  : 2;  // [27:26]
  uint32_t output_negate     : 1;  // [28]
  uint32_t saturate          : 1;  // [29]
  uint32_t transpose_a       : 1;  // [30]
  uint32_t transpose_b       : 1;  // [31]
} __attribute__((packed));
static_assert(sizeof(i_descriptor_t) == 4, "i_descriptor_t must be 32 bits");

enum class i_descriptor_kind_t : uint8_t {
  F16        = 0,
  TF32       = 1,
  BF16       = 2,
  F8F6F4     = 3,
  MXF8F6F4   = 4,
  MXF4       = 5,
  MXF4NVF4   = 6,
  I8         = 7,
  F8F16      = 8,
  F16F8      = 9,
};

// Map (At, Bt) type tags from tensor_cfg.h to PTX `.kind` encoding.
// Specialise per supported precision pair; unspecialised pairs trigger
// static_assert at use-site.
template <typename At, typename Bt>
struct i_descriptor_kind_for {
  static_assert(sizeof(At) == 0,
                "Unsupported tcgen05.mma type pair; specialise i_descriptor_kind_for<At,Bt>");
};
template <> struct i_descriptor_kind_for<fp16, fp16> { static constexpr i_descriptor_kind_t value = i_descriptor_kind_t::F16; };
template <> struct i_descriptor_kind_for<bf16, bf16> { static constexpr i_descriptor_kind_t value = i_descriptor_kind_t::BF16; };
template <> struct i_descriptor_kind_for<fp32, fp32> { static constexpr i_descriptor_kind_t value = i_descriptor_kind_t::TF32; };
template <> struct i_descriptor_kind_for<fp8,  fp8 > { static constexpr i_descriptor_kind_t value = i_descriptor_kind_t::F8F6F4; };
template <> struct i_descriptor_kind_for<fp8,  fp16> { static constexpr i_descriptor_kind_t value = i_descriptor_kind_t::F8F16; };
template <> struct i_descriptor_kind_for<fp16, fp8 > { static constexpr i_descriptor_kind_t value = i_descriptor_kind_t::F16F8; };
template <> struct i_descriptor_kind_for<int8, int8> { static constexpr i_descriptor_kind_t value = i_descriptor_kind_t::I8; };

enum class operand_fmt_code : uint32_t {
  FP32 = 0,
  FP16 = 1,
  FP8  = 2,
};

template <typename T>
struct operand_fmt_code_for {
  static_assert(sizeof(T) == 0,
                "Unsupported tcgen05.mma C/D type; specialise operand_fmt_code_for<T>");
};
template <> struct operand_fmt_code_for<fp32> { static constexpr operand_fmt_code value = operand_fmt_code::FP32; };
template <> struct operand_fmt_code_for<fp16> { static constexpr operand_fmt_code value = operand_fmt_code::FP16; };
template <> struct operand_fmt_code_for<fp8 > { static constexpr operand_fmt_code value = operand_fmt_code::FP8;  };

template <typename Ct, typename Dt>
inline __attribute__((always_inline)) constexpr uint32_t make_operand_fmt_cd() {
  return (static_cast<uint32_t>(operand_fmt_code_for<Ct>::value) & 0x3u)
       | ((static_cast<uint32_t>(operand_fmt_code_for<Dt>::value) & 0x3u) << 2);
}

// 32 B operand block (Vortex-private, lives in shared memory / LMEM).
// Pointed to by TCU_MMA rs2.
struct operand_block_t {
  uint32_t d_taddr;        // PTX d_taddr (D output and optional D input)
  uint32_t a_taddr;        // PTX a_taddr (A matrix TMEM address)
  uint32_t b_sdesc_lo;     // low 32b of 64-bit b_sdesc (PTX §9.7.16.2)
  uint32_t b_sdesc_hi;     // high 32b of 64-bit b_sdesc
  uint16_t lanes_off;      // cta_group::2 cross-CTA lane offset
  uint16_t reserved0;
  uint32_t fmt_cd;         // [1:0]=fmt_c, [3:2]=fmt_d; 0=fp32, 1=fp16, 2=fp8
  uint32_t reserved1[2];
} __attribute__((packed));
static_assert(sizeof(operand_block_t) == 32, "operand_block_t must be 32 bytes");

// 32 B cpabulk_tensor transfer args (Vortex-private, lives in LMEM).
// Pointed to by CPABULK_TENSOR_LD/ST rs2.
struct cpabulk_transfer_args_t {
  uint32_t smem_addr;       // LMEM destination (LD) or source (ST)
  uint32_t mbar_addr;       // mbarrier object address (only used if qualifier[5]=1)
  int32_t  coords[5];       // tensor coordinates (first dim_count entries valid)
  uint32_t reserved;
} __attribute__((packed));
static_assert(sizeof(cpabulk_transfer_args_t) == 32, "cpabulk_transfer_args_t must be 32 bytes");

// 8 B mbarrier object (PTX §7.6.1). Lives in shared memory / LMEM.
struct mbarrier_state_t {
  uint32_t expected_arrival_count : 20;  // [19:0]
  uint32_t pending_tx_count       : 12;  // [31:20]
  uint32_t pending_arrival_count  : 30;  // [61:32]
  uint32_t phase                  : 2;   // [63:62]
} __attribute__((packed));
static_assert(sizeof(mbarrier_state_t) == 8, "mbarrier_state_t must be 8 bytes");

// 128 B CUtensorMap-aligned descriptor for cp.async.bulk.tensor (PTX §6.4.10.4).
// Lives in DRAM constant area; host runtime prepares it.
struct tensor_map_t {
  uint64_t global_address;
  uint64_t box_size[5];
  uint64_t global_stride[5];
  uint32_t element_strides[5];
  uint8_t  element_type;
  uint8_t  interleave;
  uint8_t  swizzle;
  uint8_t  l2_promotion;
  uint8_t  oob_fill;
  uint8_t  rank;
  uint8_t  reserved0[2];
  uint32_t reserved1[3];
} __attribute__((packed));
static_assert(sizeof(tensor_map_t) == 128, "tensor_map_t must be 128 bytes");

// ----- Factories ------------------------------------------------------------

// make_i_descriptor — returns the 32-bit packed idesc value to pass as TCU_MMA rs1.
//   At, Bt: input matrix type tags (fp16, bf16, fp32, fp8, int8 from tensor_cfg.h)
//   Ct, Dt: accumulator / output type tags are not encoded in idesc.kind.
//           Use make_operand_block<Ct, Dt>() to carry fmt_c/fmt_d in operand_block_t.
//   M, N  : tile dimensions (matched to NVIDIA shape_m/shape_n encoding)
template <typename At, typename Bt, typename Ct = At, typename Dt = Ct>
inline __attribute__((always_inline)) constexpr uint32_t make_i_descriptor(
    uint16_t M,
    uint16_t N,
    bool saturate          = false,
    bool transpose_a       = false,
    bool transpose_b       = false,
    bool output_negate     = false,
    uint8_t sparsity_kind  = 0,
    uint8_t sparsity_meta_sel = 0,
    uint8_t a_storage_layout = 0,
    uint8_t b_storage_layout = 0) {
  static_cast<void>(sizeof(Ct));  // suppress unused-template-parameter warning
  static_cast<void>(sizeof(Dt));
  uint32_t v = 0;
  v |= (static_cast<uint32_t>(sparsity_meta_sel) & 0x3u) << 0;
  v |= (static_cast<uint32_t>(sparsity_kind)     & 0x1u) << 2;
  v |= (static_cast<uint32_t>(i_descriptor_kind_for<At, Bt>::value) & 0xFu) << 4;
  v |= (static_cast<uint32_t>(M) & 0xFFu) << 8;
  v |= (static_cast<uint32_t>(N) & 0xFFu) << 16;
  v |= (static_cast<uint32_t>(a_storage_layout) & 0x3u) << 24;
  v |= (static_cast<uint32_t>(b_storage_layout) & 0x3u) << 26;
  v |= (static_cast<uint32_t>(output_negate ? 1u : 0u))  << 28;
  v |= (static_cast<uint32_t>(saturate      ? 1u : 0u))  << 29;
  v |= (static_cast<uint32_t>(transpose_a   ? 1u : 0u))  << 30;
  v |= (static_cast<uint32_t>(transpose_b   ? 1u : 0u))  << 31;
  return v;
}

// make_operand_block — populate a 32B LMEM operand_block_t.
//   d_taddr   : D matrix TMEM address; callers may also set it via tcu_mma()
//   a_taddr   : A matrix TMEM address (output of tmem_alloc + offset)
//   b_sdesc   : 64-bit shared-memory descriptor for B (PTX §9.7.16.2)
//   lanes_off : cta_group::2 cross-CTA lane offset (single CTA: 0)
//   Ct, Dt    : accumulator-input and D-output external formats (fp32/fp16/fp8)
template <typename Ct = fp32, typename Dt = Ct>
inline __attribute__((always_inline)) constexpr operand_block_t make_operand_block(
    uint32_t a_taddr,
    uint64_t b_sdesc,
    uint16_t lanes_off = 0) {
  return operand_block_t{
    0,
    a_taddr,
    static_cast<uint32_t>(b_sdesc & 0xffffffffu),
    static_cast<uint32_t>(b_sdesc >> 32),
    lanes_off,
    0,
    make_operand_fmt_cd<Ct, Dt>(),
    {0, 0}
  };
}

// make_cpabulk_args — populate a 32B LMEM cpabulk_transfer_args_t.
//   smem_addr : LMEM destination (LD) or source (ST) byte address
//   mbar_addr : LMEM mbarrier object address (only consumed if qualifier[5]=1)
//   c0..c4    : tensor coordinates (first `dim_count` are valid; rest 0)
inline __attribute__((always_inline)) constexpr cpabulk_transfer_args_t make_cpabulk_args(
    uint32_t smem_addr,
    uint32_t mbar_addr = 0,
    int32_t  c0 = 0,
    int32_t  c1 = 0,
    int32_t  c2 = 0,
    int32_t  c3 = 0,
    int32_t  c4 = 0) {
  return cpabulk_transfer_args_t{
    smem_addr,
    mbar_addr,
    {c0, c1, c2, c3, c4},
    0
  };
}

inline __attribute__((always_inline)) uint16_t tmem_taddr_lane(uint32_t taddr) {
  return taddr & 0xffffu;
}

inline __attribute__((always_inline)) uint16_t tmem_taddr_col_byte(uint32_t taddr) {
  return (taddr >> 16) & 0xffffu;
}

// ============================================================================
// custom-1 (RISCV_CUSTOM1, 0x2B): tcgen05 TMEM management + cp.async.bulk.tensor
// ============================================================================
//
// Encoding: .insn r opcode, funct3, funct7=qualifier, rd, rs1, rs2
//   funct3=0b001 TMEM_REL_PERMIT
//   funct3=0b010 TMEM_ALLOC
//   funct3=0b011 TMEM_DEALLOC
//   funct3=0b100 TMEM_CP
//   funct3=0b101 TMEM_SHIFT
//   funct3=0b110 CPABULK_TENSOR_LD
//   funct3=0b111 CPABULK_TENSOR_ST

inline __attribute__((always_inline)) uint32_t tmem_alloc(uint32_t col_span, uint32_t reserved_operand) {
  uint32_t taddr;
  // funct3=0b010, qualifier[0]=cta_group::1
  __asm__ volatile (".insn r %3, 0x2, 0x0, %0, %1, %2"
    : "=r"(taddr)
    : "r"(col_span), "r"(reserved_operand), "i"(RISCV_CUSTOM1)
    : "memory");
  return taddr;
}

inline __attribute__((always_inline)) uint32_t tmem_alloc(uint32_t col_span) {
  return tmem_alloc(col_span, tmem_alloc_reserved_operand);
}

inline __attribute__((always_inline)) void tmem_dealloc(uint32_t taddr, uint32_t n_cols = 0) {
  // funct3=0b011, qualifier[0]=cta_group::1
  __asm__ volatile (".insn r %2, 0x3, 0x0, x0, %0, %1"
    :
    : "r"(taddr), "r"(n_cols), "i"(RISCV_CUSTOM1)
    : "memory");
}

inline __attribute__((always_inline)) void tmem_rel_permit() {
  // funct3=0b001, qualifier[0]=cta_group::1
  __asm__ volatile (".insn r %0, 0x1, 0x0, x0, x0, x0"
    :
    : "i"(RISCV_CUSTOM1)
    : "memory");
}

// tcgen05.cp — shared->TMEM copy.
template <uint32_t Qualifier>
inline __attribute__((always_inline)) void tmem_cp_q(uint32_t taddr, uint32_t s_desc) {
  static_assert(Qualifier < 128, "tcgen05.cp qualifier must fit in funct7");
  // funct3=0b100, qualifier[0]=cta_group::1, [3:1]=shape, [5:4]=decompress, [6]=multicast
  __asm__ volatile (".insn r %2, 0x4, %3, x0, %0, %1"
    :
    : "r"(taddr), "r"(s_desc), "i"(RISCV_CUSTOM1), "i"(Qualifier)
    : "memory");
}

inline __attribute__((always_inline)) void tmem_cp(uint32_t taddr, uint32_t s_desc) {
  tmem_cp_q<0>(taddr, s_desc);
}

template <uint32_t Shape, uint32_t Decompress = 0, bool CtaGroup = false, bool Multicast = false>
inline __attribute__((always_inline)) void tmem_cp_shape(uint32_t taddr, uint32_t s_desc) {
  static_assert(Shape < 8, "tcgen05.cp shape must fit in qualifier[3:1]");
  static_assert(Decompress < 4, "tcgen05.cp decompress must fit in qualifier[5:4]");
  constexpr uint32_t qualifier = (CtaGroup ? 1u : 0u)
                               | ((Shape & 0x7u) << 1)
                               | ((Decompress & 0x3u) << 4)
                               | (Multicast ? (1u << 6) : 0u);
  tmem_cp_q<qualifier>(taddr, s_desc);
}

inline __attribute__((always_inline)) uint32_t tmem_shift_ctl(uint32_t taddr, uint32_t control) {
  uint32_t async_id;
  // funct3=0b101 in custom-1
  __asm__ volatile (".insn r %3, 0x5, 0x0, %0, %1, %2"
    : "=r"(async_id)
    : "r"(taddr), "r"(control), "i"(RISCV_CUSTOM1)
    : "memory");
  return async_id;
}

inline __attribute__((always_inline)) uint32_t tmem_shift(uint32_t taddr) {
  return tmem_shift_ctl(taddr, 0);
}

// cp.async.bulk.tensor — DRAM -> shared.
inline __attribute__((always_inline)) uint32_t cpabulk_tensor_ld(uint32_t tensor_map_addr, uint32_t coords_ctl) {
  uint32_t async_id;
  // funct3=0b110, qualifier[2:0]=dim_count-1, [3]=im2col/tile, [4]=multicast, [5]=mbar_complete_tx
  __asm__ volatile (".insn r %3, 0x6, 0x0, %0, %1, %2"
    : "=r"(async_id)
    : "r"(tensor_map_addr), "r"(coords_ctl), "i"(RISCV_CUSTOM1)
    : "memory");
  return async_id;
}

// cp.async.bulk.tensor.mbarrier::complete_tx::bytes variant.
// qualifier[5]=1 wires the cpabulk completion to mbarrier_complete_tx via the
// args.mbar_addr field of cpabulk_transfer_args_t. Use this with the typed
// overload (rs2 in LMEM range) to drive Phase-3.2 GAP-2.
inline __attribute__((always_inline)) uint32_t cpabulk_tensor_ld_complete_tx(uint32_t tensor_map_addr, uint32_t args_lmem_ptr) {
  uint32_t async_id;
  // funct7 = 0x20 = bit 5 set
  __asm__ volatile (".insn r %3, 0x6, 0x20, %0, %1, %2"
    : "=r"(async_id)
    : "r"(tensor_map_addr), "r"(args_lmem_ptr), "i"(RISCV_CUSTOM1)
    : "memory");
  return async_id;
}

// cp.async.bulk.tensor — shared -> DRAM.
inline __attribute__((always_inline)) uint32_t cpabulk_tensor_st(uint32_t tensor_map_addr, uint32_t coords_ctl) {
  uint32_t async_id;
  __asm__ volatile (".insn r %3, 0x7, 0x0, %0, %1, %2"
    : "=r"(async_id)
    : "r"(tensor_map_addr), "r"(coords_ctl), "i"(RISCV_CUSTOM1)
    : "memory");
  return async_id;
}

// ============================================================================
// custom-2 (RISCV_CUSTOM2, 0x5B): tcgen05 sync + full mbarrier
// ============================================================================
//
// Encoding: .insn r opcode, funct3, funct7=qualifier, rd, rs1, rs2
//   funct3=0b000 MBAR_FENCE
//   funct3=0b001 MBAR_COMMIT
//   funct3=0b010 MBAR_INIT / INVALIDATE
//   funct3=0b011 MBAR_ARRIVE
//   funct3=0b100 MBAR_EXPECT_TX
//   funct3=0b101 MBAR_COMPLETE_TX
//   funct3=0b110 MBAR_WAIT
//   funct3=0b111 MBAR_TEST_TRY_WAIT

inline __attribute__((always_inline)) void mbar_fence_before() {
  // funct3=0b000, qualifier[0]=0 (before_thread_sync)
  __asm__ volatile (".insn r %0, 0x0, 0x0, x0, x0, x0"
    :
    : "i"(RISCV_CUSTOM2)
    : "memory");
}

inline __attribute__((always_inline)) void mbar_fence_after() {
  // funct3=0b000, qualifier[0]=1 (after_thread_sync)
  __asm__ volatile (".insn r %0, 0x0, 0x1, x0, x0, x0"
    :
    : "i"(RISCV_CUSTOM2)
    : "memory");
}

inline __attribute__((always_inline)) uint32_t mbar_commit(uint32_t barrier_id, uint32_t cta_mask = 0) {
  uint32_t committed;
  __asm__ volatile (".insn r %2, 0x1, 0x0, %0, %1, %3"
    : "=r"(committed)
    : "r"(barrier_id), "i"(RISCV_CUSTOM2), "r"(cta_mask)
    : "memory");
  return committed;
}

inline __attribute__((always_inline)) void mbarrier_init(uint32_t barrier_id, uint32_t count) {
  // funct3=0b010, qualifier[0]=0 (init), [1]=cluster_scope=0
  __asm__ volatile (".insn r %2, 0x2, 0x0, x0, %0, %1"
    :
    : "r"(barrier_id), "r"(count), "i"(RISCV_CUSTOM2)
    : "memory");
}

inline __attribute__((always_inline)) void mbarrier_invalidate(uint32_t barrier_id) {
  // funct3=0b010, qualifier[0]=1 (invalidate)
  __asm__ volatile (".insn r %1, 0x2, 0x1, x0, %0, x0"
    :
    : "r"(barrier_id), "i"(RISCV_CUSTOM2)
    : "memory");
}

inline __attribute__((always_inline)) void mbarrier_arrive(uint32_t barrier_id) {
  // funct3=0b011, qualifier[3:0]=0
  __asm__ volatile (".insn r %1, 0x3, 0x0, x0, %0, x0"
    :
    : "r"(barrier_id), "i"(RISCV_CUSTOM2)
    : "memory");
}

inline __attribute__((always_inline)) uint32_t mbarrier_arrive_token(uint32_t barrier_id) {
  uint32_t phase_token;
  // funct3=0b011, qualifier[3:0]=0; rd receives the observed phase token.
  __asm__ volatile (".insn r %2, 0x3, 0x0, %0, %1, x0"
    : "=r"(phase_token)
    : "r"(barrier_id), "i"(RISCV_CUSTOM2)
    : "memory");
  return phase_token;
}

inline __attribute__((always_inline)) void mbarrier_arrive_drop(uint32_t barrier_id) {
  // funct3=0b011, qualifier[1]=arrive_drop
  __asm__ volatile (".insn r %1, 0x3, 0x2, x0, %0, x0"
    :
    : "r"(barrier_id), "i"(RISCV_CUSTOM2)
    : "memory");
}

inline __attribute__((always_inline)) void mbarrier_arrive_expect_tx(uint32_t barrier_id, uint32_t tx_count) {
  // funct3=0b011, qualifier[3]=expect_tx_combo
  __asm__ volatile (".insn r %2, 0x3, 0x8, x0, %0, %1"
    :
    : "r"(barrier_id), "r"(tx_count), "i"(RISCV_CUSTOM2)
    : "memory");
}

inline __attribute__((always_inline)) void mbarrier_expect_tx(uint32_t barrier_id, uint32_t tx_count) {
  // funct3=0b100
  __asm__ volatile (".insn r %2, 0x4, 0x0, x0, %0, %1"
    :
    : "r"(barrier_id), "r"(tx_count), "i"(RISCV_CUSTOM2)
    : "memory");
}

inline __attribute__((always_inline)) void mbarrier_complete_tx(uint32_t barrier_id, uint32_t tx_count) {
  // funct3=0b101
  __asm__ volatile (".insn r %2, 0x5, 0x0, x0, %0, %1"
    :
    : "r"(barrier_id), "r"(tx_count), "i"(RISCV_CUSTOM2)
    : "memory");
}

inline __attribute__((always_inline)) void mbarrier_wait(uint32_t barrier_id, uint32_t phase_token = 0) {
  // funct3=0b110 (blocking, no timeout)
  __asm__ volatile (".insn r %2, 0x6, 0x0, x0, %0, %1"
    :
    : "r"(barrier_id), "r"(phase_token), "i"(RISCV_CUSTOM2)
    : "memory");
}

inline __attribute__((always_inline)) uint32_t mbarrier_test_wait(uint32_t barrier_id, uint32_t phase_token = 0) {
  uint32_t status;
  // funct3=0b111, qualifier[0]=0 (test_wait, non-blocking)
  __asm__ volatile (".insn r %3, 0x7, 0x0, %0, %1, %2"
    : "=r"(status)
    : "r"(barrier_id), "r"(phase_token), "i"(RISCV_CUSTOM2)
    : "memory");
  return status;
}

inline __attribute__((always_inline)) uint32_t mbarrier_try_wait(uint32_t barrier_id,
                                                                  uint32_t phase_token = 0,
                                                                  uint32_t timeout_bucket = 0) {
  uint32_t status;
  // funct3=0b111, qualifier[0]=1 (try_wait), [6:2]=timeout_bucket
  // GAS requires literal funct7; use a switch over common buckets.
  if (timeout_bucket == 0) {
    __asm__ volatile (".insn r %3, 0x7, 0x1, %0, %1, %2"
      : "=r"(status)
      : "r"(barrier_id), "r"(phase_token), "i"(RISCV_CUSTOM2)
      : "memory");
  } else {
    // Approximate: bake the LSB only; broader buckets need template/codegen.
    __asm__ volatile (".insn r %3, 0x7, 0x5, %0, %1, %2"
      : "=r"(status)
      : "r"(barrier_id), "r"(phase_token), "i"(RISCV_CUSTOM2)
      : "memory");
  }
  return status;
}

// ============================================================================
// custom-3 (RISCV_CUSTOM3, 0x7B): tcgen05 compute family
// ============================================================================
//
// Encoding: .insn r opcode, funct3, funct7=qualifier, rd, rs1, rs2
//   funct3=0b000 TCU_MMA
//   funct3=0b001 TCU_LD
//   funct3=0b010 TCU_ST
//   funct3=0b011 TCU_WAIT_LD
//   funct3=0b100 TCU_WAIT_ST
//
// TCU_MMA compact qualifier:
//   funct7[0]   enable-input-d
//   funct7[1]   .ws
//   funct7[2]   .sp
//   funct7[3]   .cta_group::2
//   funct7[5:4] collector_a_state: fill/use/lastuse/discard
//   funct7[6]   .multicast::cluster

template <uint32_t Qualifier>
inline __attribute__((always_inline)) void tcu_mma_q(uint32_t d_taddr,
                                                     uint32_t idesc,
                                                     uint32_t op_block_ptr) {
  static_assert(Qualifier < 128, "TCU_MMA qualifier must fit in funct7");
  reinterpret_cast<operand_block_t*>(op_block_ptr)->d_taddr = d_taddr;
  // rd is x0; rs1 carries idesc, rs2 points at operand_block_t.
  __asm__ volatile (".insn r %2, 0x0, %3, x0, %0, %1"
    :
    : "r"(idesc), "r"(op_block_ptr), "i"(RISCV_CUSTOM3), "i"(Qualifier)
    : "memory");
}

inline __attribute__((always_inline)) void tcu_mma(uint32_t d_taddr, uint32_t idesc, uint32_t op_block_ptr) {
  // enable-input-d=1, dense non-ws.
  tcu_mma_q<0x1>(d_taddr, idesc, op_block_ptr);
}

inline __attribute__((always_inline)) void tcu_mma_no_accum(uint32_t d_taddr, uint32_t idesc, uint32_t op_block_ptr) {
  // enable-input-d=0, dense non-ws.
  tcu_mma_q<0x0>(d_taddr, idesc, op_block_ptr);
}

// Typed convenience overload: pass a built idesc + pointer to operand_block_t.
// Phase-2 PTX-aligned form. enable_input_d defaults to 1 (D = A*B + D).
inline __attribute__((always_inline)) void tcu_mma(uint32_t d_taddr,
                                                    uint32_t idesc,
                                                    const operand_block_t* op_block) {
  tcu_mma(d_taddr, idesc, reinterpret_cast<uint32_t>(op_block));
}

inline __attribute__((always_inline)) void tcu_mma_no_accum(uint32_t d_taddr,
                                                             uint32_t idesc,
                                                             const operand_block_t* op_block) {
  tcu_mma_no_accum(d_taddr, idesc, reinterpret_cast<uint32_t>(op_block));
}

// cpabulk_tensor_ld typed overload: takes tensor_map address and args pointer.
inline __attribute__((always_inline)) uint32_t cpabulk_tensor_ld(const tensor_map_t* tmap,
                                                                  const cpabulk_transfer_args_t* args) {
  return cpabulk_tensor_ld(reinterpret_cast<uint32_t>(tmap),
                           reinterpret_cast<uint32_t>(args));
}

inline __attribute__((always_inline)) uint32_t cpabulk_tensor_st(const tensor_map_t* tmap,
                                                                  const cpabulk_transfer_args_t* args) {
  return cpabulk_tensor_st(reinterpret_cast<uint32_t>(tmap),
                           reinterpret_cast<uint32_t>(args));
}

// tmem_cp typed overload: rs2 = LMEM ptr to 64-bit s_desc storage.
inline __attribute__((always_inline)) void tmem_cp(uint32_t taddr,
                                                    const uint64_t* s_desc_ptr) {
  tmem_cp(taddr, reinterpret_cast<uint32_t>(s_desc_ptr));
}

template <uint32_t Shape, uint32_t Decompress = 0, bool CtaGroup = false, bool Multicast = false>
inline __attribute__((always_inline)) void tmem_cp_shape(uint32_t taddr,
                                                          const uint64_t* s_desc_ptr) {
  tmem_cp_shape<Shape, Decompress, CtaGroup, Multicast>(
      taddr, reinterpret_cast<uint32_t>(s_desc_ptr));
}

inline __attribute__((always_inline)) uint32_t tcu_ld(uint32_t taddr) {
  uint32_t data;
  __asm__ volatile (".insn r %2, 0x1, 0x0, %0, %1, x0"
    : "=r"(data)
    : "r"(taddr), "i"(RISCV_CUSTOM3)
    : "memory");
  return data;
}

inline __attribute__((always_inline)) void tcu_st(uint32_t taddr, uint32_t value) {
  __asm__ volatile (".insn r %2, 0x2, 0x0, x0, %0, %1"
    :
    : "r"(taddr), "r"(value), "i"(RISCV_CUSTOM3)
    : "memory");
}

inline __attribute__((always_inline)) void tcu_wait_ld() {
  __asm__ volatile (".insn r %0, 0x3, 0x0, x0, x0, x0"
    :
    : "i"(RISCV_CUSTOM3)
    : "memory");
}

inline __attribute__((always_inline)) void tcu_wait_st() {
  __asm__ volatile (".insn r %0, 0x4, 0x0, x0, x0, x0"
    :
    : "i"(RISCV_CUSTOM3)
    : "memory");
}

// NOTE: The following intrinsics from the previous Vortex ISA have been
// REMOVED in this revision:
//   mma_load_mem / mma_load_a_slot / mma_load_b_slot / mma_load_c_slot
//   mma_store_mem / mma_store_c_slot
//   tma_load / tma_load_ctl / tma_store / tma_store_ctl / tma_wait
//   tc_set_desc / tc_commit_scope / mma_sync (both fragment-based variants)
// These were Vortex-private macros not present in NVIDIA tcgen05 PTX. The
// new ISA exposes only TCU_MMA (custom-3.000) which internally fans out
// the equivalent fill-then-compute-then-drain microarchitecture.

namespace detail {

  template <typename F, std::size_t... Is>
  __attribute__((always_inline))
  constexpr void unroll_for_impl(std::index_sequence<Is...>, F&& f) {
    (f(std::integral_constant<std::size_t, Is>{}), ...);
  }

  template <std::size_t N, typename F>
  __attribute__((always_inline))
  constexpr void unroll_for(F&& f) {
    unroll_for_impl(std::make_index_sequence<N>{}, std::forward<F>(f));
  }

  template <typename T>
  struct raw_unsigned {
    using type = std::conditional_t<(sizeof(T) == 1), uint8_t,
      std::conditional_t<(sizeof(T) == 2), uint16_t,
        std::conditional_t<(sizeof(T) == 4), uint32_t,
          uint64_t>>>;
  };
  template <typename T>
  using raw_unsigned_t = typename raw_unsigned<T>::type;

  template <typename T, typename D>
  struct data_accessor_t {
    using Type = typename T::dtype;

    static inline D bit_fill(Type src) {
      static_assert(sizeof(D) % sizeof(Type) == 0, "D must be a multiple of Type in size");
      if constexpr (std::is_same_v<Type, D>) {
        return src; // passthrough
      } else {
        constexpr uint32_t count = sizeof(D) / sizeof(Type);
        constexpr uint32_t bits = 8 * sizeof(Type);
        using US = raw_unsigned_t<Type>;
        using UD = raw_unsigned_t<D>;
        auto src_u = *reinterpret_cast<const US*>(&src); // unsigned cast
        auto src_d = static_cast<UD>(src_u); // zero-extend
        UD result_u(0);
        for (uint32_t i = 0; i < count; i++) {
          result_u |= (src_d << (i * bits));
        }
        return *reinterpret_cast<const D*>(&result_u);
      }
    }

    static inline D pack_row(const Type *base, uint32_t ldm) {
      static_assert(sizeof(D) % sizeof(Type) == 0, "D must be a multiple of Type in size");
      constexpr uint32_t count = sizeof(D) / sizeof(Type);
      constexpr uint32_t bits = 8 * sizeof(Type);
      using US = raw_unsigned_t<Type>;
      using UD = raw_unsigned_t<D>;
      UD result_u(0);
      for (uint32_t i = 0; i < count; ++i) {
        auto src_u = *reinterpret_cast<const US*>(base); // unsigned cast
        auto src_d = static_cast<UD>(src_u); // zero-extend
        result_u |= (src_d << (i * bits));
        base += ldm; // next row
      }
      return *reinterpret_cast<const D*>(&result_u);
    }
  };

  template <typename D>
  struct data_accessor_t<int4, D> {

    static inline D bit_fill(uint8_t src) {
      constexpr uint32_t count = sizeof(D);
      assert((src & 0xf0) == 0 && "src must be a 4-bit value");
      using UD = raw_unsigned_t<D>;
      uint8_t src_u8 = (src << 4) | src; // pack 2 nibbles
      auto src_d = static_cast<UD>(src_u8); // zero-extend
      UD result_u(0);
      for (uint32_t i = 0; i < count; i++) {
        result_u |= (src_d << (i * 8));
      }
      return *reinterpret_cast<const D*>(&result_u);
    }
  };

  template <typename D>
  struct data_accessor_t<uint4, D> {

    static inline D bit_fill(uint8_t src) {
      constexpr uint32_t count = sizeof(D);
      assert((src & 0xf0) == 0 && "src must be a 4-bit value");
      using UD = raw_unsigned_t<D>;
      uint8_t src_u8 = (src << 4) | src; // pack 2 nibbles
      auto src_d = static_cast<UD>(src_u8); // zero-extend
      UD result_u(0);
      for (uint32_t i = 0; i < count; i++) {
        result_u |= (src_d << (i * 8));
      }
      return *reinterpret_cast<const D*>(&result_u);
    }
  };
}

template <uint32_t NT, // number of threads per warp
          typename At, // input type A
          typename Bt, // input type B
          typename Ot> // output type (C,D)
struct wmma_context_ab {
private:
  using cfg = wmma_ab_config_t<NT, At, Bt, Ot>;

  enum frag_use_t { matrix_a, matrix_b, accumulator };

  using vreg_t = float;

  template <frag_use_t U, typename T, uint32_t N>
  struct fragment_t {
    using Type = T;
    static constexpr frag_use_t Use = U;
    static constexpr uint32_t NR = N;
    std::array<vreg_t, N> data;
  };

public:

  using input_a_t = typename At::dtype;
  using input_b_t = typename Bt::dtype;
  using output_t = typename Ot::dtype;

  using input_a_accessor_t = detail::data_accessor_t<At, vreg_t>;
  using input_b_accessor_t = detail::data_accessor_t<Bt, vreg_t>;
  using output_acessor_t = detail::data_accessor_t<Ot, vreg_t>;

  static constexpr uint32_t input_a_is_subbyte = (At::bits < 8);
  static constexpr uint32_t input_b_is_subbyte = (Bt::bits < 8);

  static constexpr uint32_t a_i_ratio = cfg::a_i_ratio;
  static constexpr uint32_t b_i_ratio = cfg::b_i_ratio;
  static constexpr uint32_t tileM = cfg::tileM;
  static constexpr uint32_t tileN = cfg::tileN;
  static constexpr uint32_t tileK_a = cfg::tileK_a;
  static constexpr uint32_t tileK_b = cfg::tileK_b;

  using fragment_a   = fragment_t<matrix_a, input_a_t, cfg::NRA>;
  using fragment_b   = fragment_t<matrix_b, input_b_t, cfg::NRB>;
  using fragment_acc = fragment_t<accumulator, output_t, cfg::NRC>;

  template <typename Frag, typename T>
  static __attribute__((always_inline)) void fill_fragment(Frag &dst, T value) {
    vreg_t fill_data;
    if constexpr (Frag::Use == accumulator) {
      fill_data = output_acessor_t::bit_fill(value);
    } else if constexpr (Frag::Use == matrix_a) {
      fill_data = input_a_accessor_t::bit_fill(value);
    } else {
      fill_data = input_b_accessor_t::bit_fill(value);
    }
    detail::unroll_for<Frag::NR>([&](auto r) {
      vreg_t tmp;
      __asm__ volatile("fmv.s %0, %1" : "=f"(tmp): "f"(fill_data));
      dst.data[r] = tmp;
    });
  }

  template <mem_layout src_layout = row_major, typename Frag>
  static __attribute__((always_inline)) void load_matrix_sync(Frag &dst, const void *src, size_t ldm) {
    uint32_t lane = vx_thread_id();
    if constexpr (Frag::Use == matrix_a) {
      // Load row-major matrix A
      uint32_t block_idx = (cfg::a_block_size == NT) ? 0 : (lane / cfg::a_block_size);
      uint32_t lane_in_blk = (cfg::a_block_size == NT) ? lane : (lane % cfg::a_block_size);
      uint32_t block_row = (lane_in_blk / cfg::tcK) + (block_idx * cfg::tcM);
      uint32_t block_col = (lane_in_blk % cfg::tcK) * a_i_ratio;
      uint32_t m_stride  = cfg::a_sub_blocks * cfg::tcM;
      uint32_t k_stride  = cfg::tcK * a_i_ratio;
      if constexpr (src_layout == col_major) {
        std::swap(block_row, block_col);
      }
      auto base = reinterpret_cast<const input_a_t*>(src) + block_row * ldm + block_col;
      detail::unroll_for<Frag::NR>([&](auto r) {
        uint32_t block_m  = r / cfg::k_steps;
        uint32_t block_k  = r % cfg::k_steps;
        uint32_t elem_row = block_m * m_stride;
        uint32_t elem_col = block_k * k_stride;
        if constexpr (src_layout == col_major) {
          static_assert(input_a_is_subbyte == false, "col_major layout is not supported for sub-byte matrix_a");
          std::swap(elem_row, elem_col);
          auto ptr = base + elem_row * ldm + elem_col;
          if constexpr (sizeof(vreg_t) == sizeof(input_a_t) && !input_a_is_subbyte) {
            dst.data[r] = *reinterpret_cast<const vreg_t*>(ptr);
          } else {
            dst.data[r] = input_a_accessor_t::pack_row(ptr, ldm);
          }
        } else {
          // raw_major layout
          auto ptr = base + elem_row * ldm + elem_col;
          assert(reinterpret_cast<uintptr_t>(ptr) % alignof(vreg_t) == 0 && "pointer must be aligned to 4 bytes");
          dst.data[r] = *reinterpret_cast<const vreg_t *>(ptr);
        }
      });
    } else if constexpr (Frag::Use == matrix_b) {
      // Load column-major matrix B
      uint32_t block_idx = (cfg::b_block_size == NT) ? 0 : (lane / cfg::b_block_size);
      uint32_t lane_in_blk = (cfg::b_block_size == NT) ? lane : (lane % cfg::b_block_size);
      uint32_t block_col = (lane_in_blk / cfg::tcK) + (block_idx * cfg::tcN);
      uint32_t block_row = (lane_in_blk % cfg::tcK) * b_i_ratio;
      uint32_t n_stride  = cfg::b_sub_blocks * cfg::tcN;
      uint32_t k_stride  = cfg::tcK * b_i_ratio;
      if constexpr (src_layout == col_major) {
        std::swap(block_row, block_col);
      }
      auto base = reinterpret_cast<const input_b_t*>(src) + block_row * ldm + block_col;
      detail::unroll_for<Frag::NR>([&](auto r) {
        uint32_t block_k = r / cfg::b_sub_steps;
        uint32_t block_n = r % cfg::b_sub_steps;
        uint32_t elem_row = block_k * k_stride;
        uint32_t elem_col = block_n * n_stride;
        if constexpr (src_layout == row_major) {
          static_assert(input_b_is_subbyte == false, "row_major layout is not supported for sub-byte matrix_b");
          auto ptr = base + elem_row * ldm + elem_col;
          if constexpr (sizeof(vreg_t) == sizeof(input_b_t) && !input_b_is_subbyte) {
            dst.data[r] = *reinterpret_cast<const vreg_t*>(ptr);
          } else {
            dst.data[r] = input_b_accessor_t::pack_row(ptr, ldm);
          }
        } else {
          // col_major layout
          std::swap(elem_row, elem_col);
          auto ptr = base + elem_row * ldm + elem_col;
          assert(reinterpret_cast<uintptr_t>(ptr) % alignof(vreg_t) == 0 && "pointer must be aligned to 4 bytes");
          dst.data[r] = *reinterpret_cast<const vreg_t *>(ptr);
        }
      });
    } else {
      // Load accumulator matrix C
      if constexpr (std::is_same_v<output_t, uint16_t>) {
        // FP16 accumulator is packed as 2x16-bit values in one 32-bit vreg.
        uint32_t tcN_pairs = cfg::tcN / 2;
        uint32_t block_row = lane / tcN_pairs;
        uint32_t block_col = (lane % tcN_pairs) * 2;
        uint32_t m_stride = cfg::tcM;
        uint32_t n_stride = cfg::tcN;
        if constexpr (src_layout == col_major) {
          std::swap(block_row, block_col);
        }
        auto base = reinterpret_cast<const output_t*>(src) + block_row * ldm + block_col;
        detail::unroll_for<Frag::NR>([&](auto r) {
          uint32_t block_m  = r / cfg::n_steps;
          uint32_t block_n  = r % cfg::n_steps;
          uint32_t elem_row = block_m * m_stride;
          uint32_t elem_col = block_n * n_stride;
          if constexpr (src_layout == col_major) {
            std::swap(elem_row, elem_col);
          }
          auto ptr = base + elem_row * ldm + elem_col;
          uint32_t packed = static_cast<uint32_t>(ptr[0]) | (static_cast<uint32_t>(ptr[1]) << 16);
          dst.data[r] = *reinterpret_cast<const vreg_t*>(&packed);
        });
      } else {
        uint32_t block_row = lane / cfg::tcN;
        uint32_t block_col = lane % cfg::tcN;
        uint32_t m_stride = cfg::tcM;
        uint32_t n_stride = cfg::tcN;
        if constexpr (src_layout == col_major) {
          std::swap(block_row, block_col);
        }
        auto base = reinterpret_cast<const output_t*>(src) + block_row * ldm + block_col;
        detail::unroll_for<Frag::NR>([&](auto r) {
          uint32_t block_m  = r / cfg::n_steps;
          uint32_t block_n  = r % cfg::n_steps;
          uint32_t elem_row = block_m * m_stride;
          uint32_t elem_col = block_n * n_stride;
          if constexpr (src_layout == col_major) {
            std::swap(elem_row, elem_col);
          }
          auto ptr = base + elem_row * ldm + elem_col;
          if constexpr (sizeof(vreg_t) == sizeof(output_t)) {
            dst.data[r] = *reinterpret_cast<const vreg_t *>(ptr);
          } else {
            vreg_t tmp(0);
            *reinterpret_cast<output_t*>(&tmp) = *ptr;
            dst.data[r] = tmp;
          }
        });
      }
    }
  }

  template <mem_layout dst_layout = row_major, typename Frag>
  static __attribute__((always_inline)) void store_matrix_sync(void *dst, const Frag &src, size_t ldm) {
    static_assert(Frag::Use == accumulator, "only accumulator fragment can be stored");
    uint32_t lane = vx_thread_id();
    if constexpr (std::is_same_v<output_t, uint16_t>) {
      uint32_t tcN_pairs = cfg::tcN / 2;
      uint32_t block_row = lane / tcN_pairs;
      uint32_t block_col = (lane % tcN_pairs) * 2;
      uint32_t m_stride  = cfg::tcM;
      uint32_t n_stride  = cfg::tcN;
      if constexpr (dst_layout == col_major) {
        std::swap(block_row, block_col);
      }
      auto base = reinterpret_cast<output_t*>(dst) + block_row * ldm + block_col;
      detail::unroll_for<Frag::NR>([&](auto r) {
        uint32_t block_m  = r / cfg::n_steps;
        uint32_t block_n  = r % cfg::n_steps;
        uint32_t elem_row = block_m * m_stride;
        uint32_t elem_col = block_n * n_stride;
        if constexpr (dst_layout == col_major) {
          std::swap(elem_row, elem_col);
        }
        auto ptr = base + elem_row * ldm + elem_col;
        auto packed = *reinterpret_cast<const uint32_t*>(&src.data[r]);
        ptr[0] = static_cast<output_t>(packed & 0xffff);
        ptr[1] = static_cast<output_t>((packed >> 16) & 0xffff);
      });
    } else {
      uint32_t block_row = lane / cfg::tcN;
      uint32_t block_col = lane % cfg::tcN;
      uint32_t m_stride  = cfg::tcM;
      uint32_t n_stride  = cfg::tcN;
      if constexpr (dst_layout == col_major) {
        std::swap(block_row, block_col);
      }
      auto base = reinterpret_cast<output_t*>(dst) + block_row * ldm + block_col;
      detail::unroll_for<Frag::NR>([&](auto r) {
        uint32_t block_m  = r / cfg::n_steps;
        uint32_t block_n  = r % cfg::n_steps;
        uint32_t elem_row = block_m * m_stride;
        uint32_t elem_col = block_n * n_stride;
        if constexpr (dst_layout == col_major) {
          std::swap(elem_row, elem_col);
        }
        auto ptr = base + elem_row * ldm + elem_col;
        if constexpr (sizeof(vreg_t) == sizeof(output_t)) {
          *reinterpret_cast<vreg_t*>(ptr) = src.data[r];
        } else {
          vreg_t tmp(src.data[r]);
          *ptr = *reinterpret_cast<const output_t*>(&tmp);
        }
      });
    }
  }
};

template <uint32_t NT,
          typename It,
          typename Ot>
using wmma_context = wmma_context_ab<NT, It, It, Ot>;

} // namespace tensor
} // namespace vortex
