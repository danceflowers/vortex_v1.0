// idescriptor.h
//
// PTX-aligned tcgen05 / cp.async.bulk.tensor / mbarrier descriptor types.
//
// Naming aligns with NVIDIA PTX 8.7 / sm_100a:
//   - i_descriptor_t      (32-bit) -- tcgen05.mma "instruction descriptor" idesc
//                                    (PTX §9.7.16.5.4)
//   - operand_block_t     (32 B in shared memory) -- Vortex-private; carries the
//                                    tcgen05.mma operands that do not fit in one
//                                    RV R-type instruction.
//   - cpabulk_transfer_args_t (32 B in shared memory) -- Vortex-private; carries
//                                    cp.async.bulk.tensor coords (up to 5D),
//                                    smem destination, and bound mbar address
//                                    (when qualifier[5]=mbarrier::complete_tx)
//   - tensor_map_t        (128 B in DRAM constant area) -- aligns with
//                                    CUtensorMap (PTX §6.4.10.4)
//   - mbarrier_state_t    (8 B in shared memory) -- aligns with PTX §7.6.1
//                                    mbarrier object format
//
// Placement:
//   - idesc itself is a 32-bit register operand (TCU_WMMA rs1)
//   - operand_block_t and mbarrier_state_t live in Vortex LocalMem (shared mem)
//   - tensor_map_t lives in DRAM constant area, host runtime prepares it
//
// All structs are POD with explicit packed layout so the host runtime can
// generate them byte-for-byte compatible with simx and (future) RTL.

#pragma once

#include <cstdint>

namespace vortex {

// ============================================================================
// i_descriptor_t -- 32-bit tcgen05.mma instruction descriptor
// PTX 8.7 §9.7.16.5.4 "Instruction descriptor for tcgen05.mma"
// ============================================================================

// struct i_descriptor_t {
//   uint32_t sparsity_meta_sel : 2;  // [1:0]   sparse_meta_taddr offset (.sp variant)
//   uint32_t sparsity_kind     : 1;  // [2]     sparsity{2:4 / 1:4} encoding
//   //uint32_t reserved_3        : 1;  // [3]     reserved
//   //uint32_t kind              : 4;  // [7:4]   precision family
//   uint32_t ctype : 1; //[4:3]
//   uint32_t dtype : 1; //[6:5]
//   uint32_t atype : 1; //[8:7]
//   uint32_t btype : 1; //[10:9]
//   uint32_t shape_m           : 8;  // [15:8]  M dimension code
//   uint32_t shape_n           : 8;  // [23:16] N dimension code
//  // uint32_t a_storage_layout  : 2;  // [25:24] A storage layout
//  // uint32_t b_storage_layout  : 2;  // [27:26] B storage layout
//   uint32_t output_negate     : 1;  // [28]    negate D
//   uint32_t saturate          : 1;  // [29]    saturate (integer kinds)
//   uint32_t transpose_a       : 1;  // [30]    A transpose
//   uint32_t transpose_b       : 1;  // [31]    B transpose
// } __attribute__((packed));
// static_assert(sizeof(i_descriptor_t) == 4, "i_descriptor_t must be 32 bits");

struct i_descriptor_t {
  uint32_t sparsity_meta_sel : 2;  // [1:0]   保留但当前未显式使用；稀疏 metadata 在 shared memory 中的存储/选择信息
  uint32_t sparsity_kind     : 1;  // [2]     稀疏模式：表示 1:4 / 2:4 稀疏模式，具体 0/1 含义由 decoder 约定
  uint32_t saturate          : 1;  // [3]     保留但当前未显式使用；整数运算结果是否饱和
  uint32_t ctype             : 2;  // [5:4]   C 矩阵精度：
                                   //         00 = FP8
                                   //         01 = FP16
                                   //         10 = FP32
  uint32_t dtype             : 2;  // [7:6]   D 矩阵精度：
                                   //         00 = FP8
                                   //         01 = FP16
                                   //         10 = FP32
  uint32_t atype             : 2;  // [9:8]   A 矩阵精度：
                                   //         00 = FP8
                                   //         01 = FP16
  uint32_t btype             : 2;  // [11:10] B 矩阵精度：
                                   //         00 = FP8
                                   //         01 = FP16
  uint32_t negate_a          : 1;  // [12]    保留但当前未显式使用；A 矩阵是否取负
  uint32_t negate_b          : 1;  // [13]    保留但当前未显式使用；B 矩阵是否取负
  uint32_t transpose_a       : 1;  // [14]    A 矩阵是否转置
  uint32_t transpose_b       : 1;  // [15]    B 矩阵是否转置
  uint32_t shape_m           : 4;  // [19:16] A 矩阵的 M 维度编码；
                                   //         低 4 bit 隐含为 0，即 M = shape_m << 4
  uint32_t shape_n           : 4;  // [23:20] B 矩阵的 N 维度编码；
                                   //         低 4 bit 隐含为 0，即 N = shape_n << 4
  uint32_t b_shift           : 8;  // [31:24] 保留但当前未显式使用；
                                   //         WS 模式下 B 矩阵在 collector buffer 中允许的最大移位量
} __attribute__((packed));

static_assert(sizeof(i_descriptor_t) == 4, "i_descriptor_t must be 32 bits");

// kind selector (matches PTX §9.7.16.5.4 .kind encoding)
enum class i_descriptor_kind_t : uint8_t {
  F16        = 0,  // .kind::f16
  TF32       = 1,  // .kind::tf32
  BF16       = 2,  // .kind::bf16
  F8F6F4     = 3,  // .kind::f8f6f4
  MXF8F6F4   = 4,  // .kind::mxf8f6f4
  MXF4       = 5,  // .kind::mxf4
  MXF4NVF4   = 6,  // .kind::mxf4nvf4
  I8         = 7,  // .kind::i8
  F8F16      = 8,  // Vortex extension: A fp8, B fp16
  F16F8      = 9,  // Vortex extension: A fp16, B fp8
};

// ============================================================================
// operand_block_t -- 32 B in shared memory
// Vortex-private; pointed by TCU_WMMA rs2.
//
// PTX tcgen05.mma takes more operands than one RV R-type instruction can carry.
// The custom instruction keeps idesc in rs1 and points rs2 at this block.
// d_taddr lives here; rd stays x0 so the instruction has no false GPR writeback.
//
// Notes vs. NVIDIA PTX:
//   - There is NO separate C operand in tcgen05.mma. When enable_input_d=1,
//     d_taddr is read as the accumulator input AND written as D output
//     (read-modify-write at the same TMEM location). So no `c_taddr` field.
//   - A's source (TMEM addr vs. shared sdesc) is implied by idesc.kind
//     (PTX §9.7.16.5.4). For Vortex Phase-2 we model A from TMEM only;
//     a_taddr is the single A operand. Shared-sourced A (.kind=f8f6f4 et al.)
//     is left out for now and reintroduced when those kinds land.
//   - Current compact TCU_WMMA funct7 carries enable_input_d, ws, sp,
//     cta_group, collector_a_state and multicast. scale-input-d is intentionally
//     not encoded yet.
//   - lanes_off (cta_group::2 lane offset) lives here because it varies
//     per instruction instance.
//   - fmt_cd is a Vortex-private extension in the operand block. It carries
//     fmt_c/fmt_d because the compact RISC-V instruction and 4-bit idesc.kind
//     do not have enough spare bits for all C/D precision combinations.
// ============================================================================

struct operand_block_t {
  uint32_t d_taddr;            // PTX d_taddr (D output and optional D input)
  uint32_t a_taddr;            // PTX a_taddr (A matrix TMEM address)
  uint32_t b_sdesc_lo;         // low 32 b of 64-bit b_sdesc (PTX §9.7.16.2)
  uint32_t b_sdesc_hi;         // high 32 b of 64-bit b_sdesc
  uint16_t lanes_off;          // cta_group::2 cross-CTA lane offset
  uint16_t reserved0;       // [1:0]=bbuf_idx for ws=1 TCU_WMMA, [15:2]=reserved
  uint32_t fmt_cd;             // [1:0]=fmt_c, [3:2]=fmt_d; 0=fp32, 1=fp16, 2=fp8
  uint32_t reserved1[2];       // tail-pad to 32 B; available for future fields
} __attribute__((packed));
static_assert(sizeof(operand_block_t) == 32, "operand_block_t must be 32 bytes");

// collector buffer state encoding (PTX §9.7.16.5.5).
// Encoded in TCU_WMMA qualifier[5:4]; only meaningful when collector variants
// are enabled by the frontend.
enum class CollectorState : uint8_t {
  Fill     = 0,  // .collector::a::fill
  Use      = 1,  // .collector::a::use
  LastUse  = 2,  // .collector::a::lastuse
  Discard  = 3,  // .collector::a::discard
};

// ============================================================================
// cpabulk_transfer_args_t -- 32 B in shared memory
// Vortex-private; pointed by CPABULK_TENSOR_LD/ST rs2.
// Holds cp.async.bulk.tensor coordinates (up to 5D), smem destination,
// and (for LD with qualifier[5]=mbarrier::complete_tx) mbar address.
// ============================================================================

struct cpabulk_transfer_args_t {
  uint32_t smem_addr;     // LMEM destination (LD) or source (ST)
  uint32_t mbar_addr;     // mbarrier object address (only used if qualifier[5]=1)
  int32_t  coords[5];     // tensor coordinates; first dim_count entries are valid
  uint32_t reserved;
} __attribute__((packed));
static_assert(sizeof(cpabulk_transfer_args_t) == 32, "cpabulk_transfer_args_t must be 32 bytes");

// ============================================================================
// tensor_map_t -- 128 B in DRAM (constant area)
// Aligns with CUDA driver API CUtensorMap; PTX §6.4.10.4
// ============================================================================

// CUtensorMap is opaque 128 B in CUDA driver. Vortex picks a deterministic
// internal layout that fits in 128 B and exposes the fields cp.async.bulk.tensor
// needs (rank, dims, strides, element_type, swizzle, oob_fill).
struct tensor_map_t {
  uint64_t global_address;            // [0..7]    tensor base address (DRAM)
  uint64_t box_size[5];               // [8..47]   tile size per dim
  uint64_t global_stride[5];          // [48..87]  global tensor stride per dim
  uint32_t element_strides[5];        // [88..107] element stride per dim
  uint8_t  element_type;              // [108]     CUtensorMapDataType
  uint8_t  interleave;                // [109]     CUtensorMapInterleave
  uint8_t  swizzle;                   // [110]     CUtensorMapSwizzle
  uint8_t  l2_promotion;              // [111]     CUtensorMapL2promotion
  uint8_t  oob_fill;                  // [112]     CUtensorMapFloatOOBfill
  uint8_t  rank;                      // [113]     number of dimensions (1..5)
  uint8_t  reserved0[2];              // [114..115]
  uint32_t reserved1[3];              // [116..127] tail-pad to 128
} __attribute__((packed));
static_assert(sizeof(tensor_map_t) == 128, "tensor_map_t must be 128 bytes");

// ============================================================================
// mbarrier_state_t -- 8 B in shared memory
// PTX §7.6.1 mbarrier object format. Bit-packed:
//   bits[63:62]   phase parity (2-bit; bit 63 = parity, bit 62 reserved)
//   bits[61:32]   pending arrival count (30 bits)
//   bits[31:20]   pending tx count (12 bits, in bytes/16 unit per PTX
//                                   simplified for CModel; see §10.4)
//   bits[19:0]    expected arrival count (20 bits)
// Vortex CModel splits these into a struct for clarity; the on-LMEM bit layout
// is preserved by writing/reading via a union-like helper or by stronger
// load/store routines in core.cpp.
// ============================================================================

struct mbarrier_state_t {
  uint32_t expected_arrival_count : 20;  // [19:0]
  uint32_t pending_tx_count       : 12;  // [31:20]  (byte count, see PTX §7.6.4)
  uint32_t pending_arrival_count  : 30;  // [61:32]
  uint32_t phase                  : 2;   // [63:62]
} __attribute__((packed));
static_assert(sizeof(mbarrier_state_t) == 8, "mbarrier_state_t must be 8 bytes");

// expected_tx_count is *not* stored on-mbarrier per PTX §7.6.4 (the runtime
// tracks it via mbarrier.expect_tx accumulating into a separate counter that
// shares the pending_tx_count field semantically: expect_tx adds to expected,
// complete_tx adds to completed, phase advances when both arrival and tx
// completion meet expected). For Vortex we keep one running counter
// (pending_tx_count) that is incremented by complete_tx and decremented when
// it reaches the expect_tx high-water; see Core implementation.

} // namespace vortex
