// tc_decode.h
//
// TensorCore 二级译码。Phase-2 后只处理 tensor 计算指令族（TCU_MMA / TCU_LD /
// TCU_ST）；TMEM_*, MBAR_*, CPABULK_*, TCU_WAIT_* 由 Core / execute.cpp 直接
// 处理。
//
// 设计原则（Phase-2）：
//   - Vortex Core decode.cpp 对 TCU_MMA 仅做"识别+转发"：抓 (rs1, rs2,
//     funct7=qualifier) 3 个原始字段送进来，不解释 idesc/operand_block。
//   - 本模块从 rs1（idesc 32-bit value）+ rs2（operand_block_t LMEM 指针）
//     完成完整译码：解 idesc 各位段、lmem_read 32B operand_block、综合 qualifier
//     模式位，输出 TcDecodedMmaCmd。
//   - 不再有 TC_SET_DESC / cached_descriptor 概念；每条 TCU_MMA 自带 idesc。

#pragma once

#include <cstdint>
#include "types.h"
#include "idescriptor.h"

namespace vortex {

class Core;

// Phase-2: legacy TcDecodedCmd removed; TcDecodedMmaCmd / TcDecodedLdStCmd
// (defined below) supersede it.

// ============================================================================
// TcDecodedMmaCmd —— TCU_MMA 译码结果，喂给 TensorAsyncFrontend::enqueue_async_tcu_mma
// ============================================================================
struct TcDecodedMmaCmd {
  uint32_t wid = 0;

  // PTX idesc 各字段解码结果
  uint32_t fmt_a = 0;
  uint32_t fmt_b = 0;
  uint32_t fmt_c = 0;
  uint32_t fmt_d = 0;
  uint32_t shape_m = 0;
  uint32_t shape_n = 0;
  uint8_t  sparsity_kind = 0;       // 0=dense, 1=2:4, 2=1:4
  uint8_t  saturate = 0;
  uint8_t  transpose_a = 0;
  uint8_t  transpose_b = 0;
  uint8_t  output_negate = 0;

  // qualifier 模式位
  uint8_t  enable_input_d = 0;      // qualifier[0]
  uint8_t  ws = 0;                  // qualifier[1]
  uint8_t  sp = 0;                  // qualifier[2]
  uint8_t  cta_group = 0;           // qualifier[3] (1 bit)
  uint8_t  collector_a_state = 0;   // qualifier[5:4] (fill/use/lastuse/discard)
  uint8_t  multicast = 0;           // qualifier[6]

  // operand_block_t 解出的字段（来自 LMEM）
  uint32_t a_taddr = 0;             // A matrix TMEM address
  uint64_t b_sdesc = 0;             // 64-bit b_sdesc
  uint16_t lanes_off = 0;           // cta_group::2 lane offset

  // d_taddr 来自 operand_block_t；rd 固定 x0，无 GPR 写回。
  uint32_t d_taddr = 0;
};

// ============================================================================
// TcDecodedLdStCmd —— tcgen05.{ld, st} 译码结果（占位；当前留作存根）
// ============================================================================
struct TcDecodedLdStCmd {
  uint32_t wid = 0;
  uint32_t taddr = 0;
  uint32_t shape_code = 0;          // qualifier[2:0]
  uint32_t num = 0;                 // qualifier[5:3]
  uint8_t  pack = 0;                // qualifier[6]
};

class TcDecode {
public:
  // ===== Phase-2 new API: tensor compute decoding only =====

  // tcgen05.mma 译码：rs1=idesc(32-bit), rs2=operand_block_t LMEM ptr,
  // qualifier=funct7。读 LMEM 一次拿 operand_block，结合 idesc
  // 各位段 + qualifier 模式位输出完整 TcDecodedMmaCmd。
  // 失败（operand_block 读失败 / idesc 字段非法）返回 false。
  bool decode_tcu_mma(Core* core,
                      uint32_t wid,
                      uint32_t rs1_value,
                      uint32_t rs2_value,
                      uint32_t qualifier,
                      TcDecodedMmaCmd* out) const;

  // tcgen05.ld / tcgen05.st 译码：单步 RF<->TMEM，无 LMEM 二级读。
  bool decode_tcu_ld_st(uint32_t wid,
                        uint32_t taddr,
                        uint32_t qualifier,
                        TcDecodedLdStCmd* out) const;

  void reset();
};

} // namespace vortex
