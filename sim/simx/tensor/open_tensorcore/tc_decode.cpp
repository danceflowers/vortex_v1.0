// tc_decode.cpp
//
// TensorCore coprocessor instruction decode implementation.

#include "open_tensorcore/tc_decode.h"
#include "core.h"
#include "tensor_cfg.h"

#include <cstring>

namespace vortex {

namespace vt = vortex::tensor;

namespace {

/// Map atype/btype/ctype/dtype 2-bit fields from i_descriptor_t to Vortex fmt ids.
bool idesc_to_fmts(uint32_t atype, uint32_t btype,
                   uint32_t ctype, uint32_t dtype,
                   uint32_t* fmt_a, uint32_t* fmt_b,
                   uint32_t* fmt_c, uint32_t* fmt_d) {
  auto type_to_fmt = [](uint32_t code, bool is_cd) -> uint32_t {
    switch (code & 0x3u) {
    case 0: return vt::fp8::id;                         // 00 = FP8
    case 1: return vt::fp16::id;                        // 01 = FP16
    case 2: return is_cd ? vt::fp32::id : vt::fp16::id; // 10 = FP32 (C/D only)
    default: return 0xFFFFFFFFu;                        // invalid
    }
  };
  *fmt_a = type_to_fmt(atype, false);
  *fmt_b = type_to_fmt(btype, false);
  *fmt_c = type_to_fmt(ctype, true);
  *fmt_d = type_to_fmt(dtype, true);
  return *fmt_a != 0xFFFFFFFFu && *fmt_b != 0xFFFFFFFFu
      && *fmt_c != 0xFFFFFFFFu && *fmt_d != 0xFFFFFFFFu;
}

// Convert sparsity kind + .sp qualifier -> Vortex sparse_mode.
uint8_t sparsity_to_mode(uint32_t kind_bit, bool sp_qualifier) {
  if (!sp_qualifier) return vt::sparse_none;
  return (kind_bit == 0) ? vt::sparse_2_4 : vt::sparse_1_4;
}

} // namespace

bool TcDecode::decode_tcu_mma(Core* core,
                              uint32_t wid,
                              uint32_t rs1_value,
                              uint32_t rs2_value,
                              uint32_t qualifier,
                              TcDecodedMmaCmd* out) const {
  if (out == nullptr || core == nullptr) {
    return false;
  }

  // Decode 32-bit idesc from rs1 (PTX §9.7.16.5.4).
  i_descriptor_t idesc;
  static_assert(sizeof(idesc) == sizeof(uint32_t));
  std::memcpy(&idesc, &rs1_value, sizeof(idesc));

  // qualifier modifier bits (7-bit funct7 field).
  out->wid               = wid;
  out->enable_input_d    = (qualifier >> 0) & 1;
  out->ws                = (qualifier >> 1) & 1;
  out->sp                = (qualifier >> 2) & 1;
  out->cta_group         = (qualifier >> 3) & 1;
  out->collector_a_state = (qualifier >> 4) & 0x3;
  out->multicast         = (qualifier >> 6) & 1;

  // idesc fields — precision comes from atype/btype/ctype/dtype.
  if (!idesc_to_fmts(idesc.atype, idesc.btype, idesc.ctype, idesc.dtype,
                     &out->fmt_a, &out->fmt_b, &out->fmt_c, &out->fmt_d)) {
    return false;
  }
  out->shape_m       = static_cast<uint32_t>(idesc.shape_m) << 4;
  out->shape_n       = static_cast<uint32_t>(idesc.shape_n) << 4;
  out->sparsity_kind = sparsity_to_mode(idesc.sparsity_kind,
                                        out->sp != 0);
  out->sparsity_meta_sel = idesc.sparsity_meta_sel;
  out->saturate      = idesc.saturate;
  out->transpose_a   = idesc.transpose_a;
  out->transpose_b   = idesc.transpose_b;

  // Read 32B operand_block_t from LMEM at rs2_value.
  operand_block_t op_block;
  static_assert(sizeof(op_block) == 32);
  core->lmem_read(&op_block, static_cast<uint64_t>(rs2_value), sizeof(op_block));

  out->d_taddr   = op_block.d_taddr;
  out->a_taddr   = op_block.a_taddr;
  out->b_sdesc   = (uint64_t(op_block.b_sdesc_hi) << 32) | uint64_t(op_block.b_sdesc_lo);
  out->lanes_off = op_block.lanes_off;
  // Precision (fmt_a/b/c/d) already resolved from idesc atype/btype/ctype/dtype above.

  return true;
}

bool TcDecode::decode_tcu_ld_st(uint32_t wid,
                                uint32_t taddr,
                                uint32_t qualifier,
                                TcDecodedLdStCmd* out) const {
  if (out == nullptr) return false;
  out->wid = wid;
  out->taddr = taddr;
  out->shape_code = (qualifier >> 0) & 0x7;
  out->num        = (qualifier >> 3) & 0x7;
  out->pack       = (qualifier >> 6) & 0x1;
  return true;
}

void TcDecode::reset() {
  // idesc is carried by each instruction, so reset has no descriptor state.
}

} // namespace vortex
