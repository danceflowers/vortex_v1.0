// i_descriptor.h
//
// Legacy MMA descriptor-table entry.
//
// IDescriptor is still used by the descriptor-table/TMA setup path for older
// software-visible matrix configuration records. The tcgen05.mma path should
// use idescriptor_t from idescriptor.h instead: that 32-bit value is carried as
// an instruction operand and decoded per MMA instruction.
//
// Keep this header independent from the TensorCore control decoder so legacy
// table setup does not leak into the PTX-aligned tcgen05 instruction decoder.

#pragma once

#include <cstdint>

namespace vortex {

struct IDescriptor {
  uint32_t fmt_a = 0;           // A 操作数精度格式 ID
  uint32_t fmt_b = 0;           // B 操作数精度格式 ID
  uint32_t fmt_c = 0;           // C 累加器精度格式 ID
  uint32_t fmt_d = 0;           // D 输出精度格式 ID
  uint8_t output_resident = 0;  // 输出驻留：C 累加器在 FIFO 中驻留循环累加
  uint8_t ws = 0;               // weight-stationary：BMem 首次 fill 后锁定驻留
  uint8_t sp = 0;               // 特殊精度标志
  uint8_t sparse_mode = 0;      // A 操作数稀疏模式
  uint16_t a_rows = 0;          // A 矩阵行数 (M 维)
  uint16_t a_cols = 0;          // A 矩阵列数 (K 维)
  uint16_t b_rows = 0;          // B 矩阵行数 (K 维)
  uint16_t b_cols = 0;          // B 矩阵列数 (N 维)
  uint16_t c_rows = 0;          // C 矩阵行数 (M 维)
  uint16_t c_cols = 0;          // C 矩阵列数 (N 维)
} __attribute__((packed));

} // namespace vortex
