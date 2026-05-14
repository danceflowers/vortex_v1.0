// tc_decode.cpp
//
// TensorCore coprocessor instruction decode implementation.

#include "open_tensorcore/tensor_control/tc_decode.h"
#include "core.h"
#include "tensor_cfg.h"

#include <cstring>

namespace vortex {

namespace vt = vortex::tensor;

namespace {

// PTX idesc.kind (4-bit code) -> Vortex fmt id pair (a, b, c).
// Returns false on unsupported kind.
bool kind_to_fmt(uint32_t kind, uint32_t* fmt_a, uint32_t* fmt_b,
                 uint32_t* fmt_c, uint32_t* fmt_d) {
  switch (static_cast<i_descriptor_kind_t>(kind)) {
  case i_descriptor_kind_t::F16:
    *fmt_a = vt::fp16::id; *fmt_b = vt::fp16::id;
    *fmt_c = vt::fp32::id; *fmt_d = vt::fp32::id;
    return true;
  case i_descriptor_kind_t::BF16:
    *fmt_a = vt::bf16::id; *fmt_b = vt::bf16::id;
    *fmt_c = vt::fp32::id; *fmt_d = vt::fp32::id;
    return true;
  case i_descriptor_kind_t::TF32:
    *fmt_a = vt::fp32::id; *fmt_b = vt::fp32::id;
    *fmt_c = vt::fp32::id; *fmt_d = vt::fp32::id;
    return true;
  case i_descriptor_kind_t::F8F6F4:
  case i_descriptor_kind_t::MXF8F6F4:
    *fmt_a = vt::fp8::id; *fmt_b = vt::fp8::id;
    *fmt_c = vt::fp32::id; *fmt_d = vt::fp32::id;
    return true;
  case i_descriptor_kind_t::F8F16:
    *fmt_a = vt::fp8::id; *fmt_b = vt::fp16::id;
    *fmt_c = vt::fp32::id; *fmt_d = vt::fp32::id;
    return true;
  case i_descriptor_kind_t::F16F8:
    *fmt_a = vt::fp16::id; *fmt_b = vt::fp8::id;
    *fmt_c = vt::fp32::id; *fmt_d = vt::fp32::id;
    return true;
  case i_descriptor_kind_t::I8:
    // Vortex CModel doesn't yet implement int8 path; placeholder.
    *fmt_a = 0; *fmt_b = 0; *fmt_c = 0; *fmt_d = 0;
    return true;
  default:
    return false;
  }
}

bool fmt_code_to_fmt(uint32_t code, uint32_t* fmt) {
  switch (code & 0x3u) {
  case 0: *fmt = vt::fp32::id; return true;
  case 1: *fmt = vt::fp16::id; return true;
  case 2: *fmt = vt::fp8::id;  return true;
  default:
    return false;
  }
}

bool fmt_cd_to_fmt(uint32_t fmt_cd, uint32_t* fmt_c, uint32_t* fmt_d) {
  return fmt_code_to_fmt(fmt_cd & 0x3u, fmt_c)
      && fmt_code_to_fmt((fmt_cd >> 2) & 0x3u, fmt_d);
}

// PTX idesc.sparsity_kind (1 bit + sparsity_meta_sel 2 bit) -> Vortex sparse_mode.
uint8_t sparsity_to_mode(uint32_t kind_bit, uint32_t /*meta_sel*/, bool sp_qualifier) {
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

  // idesc fields.
  if (!kind_to_fmt(idesc.kind, &out->fmt_a, &out->fmt_b, &out->fmt_c, &out->fmt_d)) {
    return false;
  }
  out->shape_m       = idesc.shape_m;
  out->shape_n       = idesc.shape_n;
  out->sparsity_kind = sparsity_to_mode(idesc.sparsity_kind,
                                        idesc.sparsity_meta_sel,
                                        out->sp != 0);
  out->sparsity_meta_sel = idesc.sparsity_meta_sel;
  out->saturate      = idesc.saturate;
  out->transpose_a   = idesc.transpose_a;
  out->transpose_b   = idesc.transpose_b;
  out->output_negate = idesc.output_negate;

  // Read 32B operand_block_t from LMEM at rs2_value.
  operand_block_t op_block;
  static_assert(sizeof(op_block) == 32);
  core->lmem_read(&op_block, static_cast<uint64_t>(rs2_value), sizeof(op_block));

  out->d_taddr   = op_block.d_taddr;
  out->a_taddr   = op_block.a_taddr;
  out->b_sdesc   = (uint64_t(op_block.b_sdesc_hi) << 32) | uint64_t(op_block.b_sdesc_lo);
  out->lanes_off = op_block.lanes_off;
  if (!fmt_cd_to_fmt(op_block.fmt_cd, &out->fmt_c, &out->fmt_d)) {
    return false;
  }

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
