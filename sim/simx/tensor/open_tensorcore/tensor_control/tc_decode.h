// tc_decode.h
//
// Phase-2 TensorCore decoder for the tensor compute instruction family.
// TMEM_*, MBAR_*, CPABULK_*, and TCU_WAIT_* are handled directly by Core /
// execute.cpp; this module decodes tcgen05.mma and the tcgen05 ld/st forms.
//
// The Vortex core decode stage only identifies TCU_MMA and forwards raw
// (rs1, rs2, qualifier) fields. TcDecode interprets rs1 as a 32-bit
// i_descriptor_t, reads the 32B operand_block_t from LMEM through rs2, applies
// qualifier mode bits, and returns a compact command for TensorUnit.

#pragma once

#include <cstdint>
#include "types.h"
#include "idescriptor.h"

namespace vortex {

class Core;

// TcDecodedMmaCmd / TcDecodedLdStCmd are the only decoded tensor compute forms.

// Decoded tcgen05.mma command consumed by TensorUnit.
struct TcDecodedMmaCmd {
  uint32_t wid = 0;

  // Fields decoded from the PTX i_descriptor_t.
  uint32_t fmt_a = 0;
  uint32_t fmt_b = 0;
  uint32_t fmt_c = 0;
  uint32_t fmt_d = 0;
  uint32_t shape_m = 0;
  uint32_t shape_n = 0;
  uint8_t  sparsity_kind = 0;       // 0=dense, 1=2:4, 2=1:4
  uint8_t  sparsity_meta_sel = 0;   // selects the 64B metadata packet next to A payload
  uint8_t  saturate = 0;
  uint8_t  transpose_a = 0;
  uint8_t  transpose_b = 0;
  uint8_t  output_negate = 0;

  // Modifier bits decoded from the qualifier/funct7 field.
  uint8_t  enable_input_d = 0;      // qualifier[0]
  uint8_t  ws = 0;                  // qualifier[1]
  uint8_t  sp = 0;                  // qualifier[2]
  uint8_t  cta_group = 0;           // qualifier[3] (1 bit)
  uint8_t  collector_a_state = 0;   // qualifier[5:4] (fill/use/lastuse/discard)
  uint8_t  multicast = 0;           // qualifier[6]

  // Fields loaded from operand_block_t in LMEM.
  uint32_t a_taddr = 0;             // A matrix TMEM address
  uint64_t b_sdesc = 0;             // 64-bit b_sdesc
  uint16_t lanes_off = 0;           // cta_group::2 lane offset

  // d_taddr also comes from operand_block_t; rd is fixed to x0.
  uint32_t d_taddr = 0;
};

// Decoded tcgen05.{ld,st} command. Kept small because these are one-step
// register-file <-> TMEM operations.
struct TcDecodedLdStCmd {
  uint32_t wid = 0;
  uint32_t taddr = 0;
  uint32_t shape_code = 0;          // qualifier[2:0]
  uint32_t num = 0;                 // qualifier[5:3]
  uint8_t  pack = 0;                // qualifier[6]
};

class TcDecode {
public:
  // Decode tcgen05.mma. rs1 carries i_descriptor_t, rs2 points to
  // operand_block_t in LMEM, and qualifier carries funct7 mode bits.
  bool decode_tcu_mma(Core* core,
                      uint32_t wid,
                      uint32_t rs1_value,
                      uint32_t rs2_value,
                      uint32_t qualifier,
                      TcDecodedMmaCmd* out) const;

  // Decode one-step tcgen05.ld/st register-file <-> TMEM commands.
  bool decode_tcu_ld_st(uint32_t wid,
                        uint32_t taddr,
                        uint32_t qualifier,
                        TcDecodedLdStCmd* out) const;

  void reset();
};

} // namespace vortex
