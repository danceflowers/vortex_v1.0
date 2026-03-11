#pragma once
#include "fp_types.h"
#include <cstdint>
#include <algorithm>

inline int clz_hardware(uint16_t val, int width) {
    if (val == 0) return width;
    int count = 0;
    for (int i = width - 1; i >= 0; --i) {
        if ((val >> i) & 1) break;
        count++;
    }
    return count;
}

// 统一的 FP9 转换函数
// 核心思想：将所有格式映射到统一的内部宽格式 (Internal Wide Format)，复用加法器和舍入逻辑
uint16_t convert_to_fp9(uint32_t raw_bits, PrecisionType prec) {
    // =========================================================
    // Stage 1: Control Logic & Input Muxing (配置与解包)
    // =========================================================
    // 这一步模拟硬件的多路选择器 (MUX)，根据模式选择信号源
    
    bool sign;
    int raw_exp;
    int raw_mant;
    

    int input_bias_offset; // 用于将输入 Bias 转换到 FP9 Bias (15) 的差值
    int mant_align_shift;  // 将尾数左对齐到 10-bit 的移位量
    int max_exp_code;      // 输入格式的最大指数编码 (用于判断 NaN/Inf)
    int input_mant_width;  

    switch (prec) {
        case PREC_FP4_E2M1:
            // FP4: 1 Sign, 2 Exp (Bias 1?), 1 Mant
            // Note: FP4 subnorm (e=0) maps to FP9 normal. Bias delta: 15 - 1 = 14.
            sign = (raw_bits >> 3) & 1;
            raw_exp = (raw_bits >> 1) & 3;
            raw_mant = raw_bits & 1;
            input_bias_offset = 14; 
            mant_align_shift = 9; // 1 bit -> 10 bits
            max_exp_code = 3;
            input_mant_width = 1;
            break;

        case PREC_FP8_E4M3:
            // FP8 E4M3: 1 Sign, 4 Exp (Bias 7), 3 Mant
            // Bias delta: 15 - 7 = 8.
            sign = (raw_bits >> 7) & 1;
            raw_exp = (raw_bits >> 3) & 0xF;
            raw_mant = raw_bits & 7;
            input_bias_offset = 8;
            mant_align_shift = 7; // 3 bits -> 10 bits
            max_exp_code = 15;
            input_mant_width = 3;
            break;

        case PREC_FP8_E5M2:
            // FP8 E5M2: 1 Sign, 5 Exp (Bias 15), 2 Mant
            // Bias delta: 15 - 15 = 0.
            sign = (raw_bits >> 7) & 1;
            raw_exp = (raw_bits >> 2) & 0x1F;
            raw_mant = raw_bits & 3;
            input_bias_offset = 0;
            mant_align_shift = 8; // 2 bits -> 10 bits
            max_exp_code = 31;
            input_mant_width = 2;
            break;

        case PREC_FP16:
            // FP16: 1 Sign, 5 Exp (Bias 15), 10 Mant
            sign = (raw_bits >> 15) & 1;
            raw_exp = (raw_bits >> 10) & 0x1F;
            raw_mant = raw_bits & 0x3FF;
            input_bias_offset = 0;
            mant_align_shift = 0;
            max_exp_code = 31;
            input_mant_width = 10;
            break;
            
        default: return 0;
    }

    // =========================================================
    // Stage 2: Special Case Detection (NaN / Inf / Zero)
    // =========================================================
    
    // 检查 NaN 或 Inf
    if (raw_exp == max_exp_code) {
        if (raw_mant != 0) return (sign << 8) | (0x1F << 3) | 4; // NaN (Canonical)
        return (sign << 8) | (0x1F << 3); // Inf
    }

    // 检查 Zero
    if (raw_exp == 0 && raw_mant == 0) {
        return (sign << 8);
    }

    // =========================================================
    // Stage 3: Unified Normalization Pipeline (核心数据通路)
    // =========================================================
    
    // 内部统一格式：Signed Exponent (Bias 15) + 11-bit Mantissa (1.xxxxxxxxx)
    int true_exp;
    uint32_t full_mant; // 包含显式或隐式的 '1'

    // 将尾数对齐到 10-bit 精度 (FP16 级别)
    uint32_t aligned_mant = raw_mant << mant_align_shift;

    if (raw_exp == 0) {
        // Input Subnormal Handling (输入非规格化处理)
        // 硬件复用：CLZ 模块
        int lz = clz_hardware(aligned_mant, 10);
        
        // 标准化：左移消除前导零
        // 尾数变成 1.xxxx 格式 (bit 10 为 1)
        full_mant = (aligned_mant << (lz + 1)) & 0x7FF; // Keep 11 bits
        
        // 指数调整：
        // Subnorm 真实指数是 (1 - InputBias).
        // 我们需要转换到 Bias 15.
        // Target Exp = (1 - InputBias) + 15 - LZ
        //            = 1 + (15 - InputBias) - LZ
        //            = 1 + input_bias_offset - LZ
        true_exp = 1 + input_bias_offset - (lz + 1); 
    } else {
        // Input Normal Handling
        // 恢复隐藏的 '1' 到 bit 10
        full_mant = (1 << 10) | aligned_mant;
        true_exp = raw_exp + input_bias_offset;
    }

    // =========================================================
    // Stage 4: Output Denormalization (输出端非规格化处理)
    // =========================================================
    // 如果计算出的指数 <= 0，说明在 FP9 格式下是 Subnormal
    
    if (true_exp <= 0) {
        // 需要右移以匹配 FP9 的最小指数 (Exp=1, stored as 0)
        // Shift amount = 1 - true_exp
        int shift_amt = 1 - true_exp;
        
        // 硬件限制：最大移位限制，防止移位过大耗费资源 (FP9 只有 3 bit 尾数，移位 > 12 全为 0)
        if (shift_amt > 12) {
            full_mant = 0; // Underflow to zero
        } else {
            // 带有 Sticky bit 的右移逻辑
            // 在硬件中，移出的位会 OR 在一起形成 Sticky
            int sticky_mask = (1 << shift_amt) - 1;
            bool sticky = (full_mant & sticky_mask) != 0;
            full_mant >>= shift_amt;
            if (sticky) full_mant |= 1; // 将 Sticky 并在 LSB
        }
        true_exp = 0;
    }

    // =========================================================
    // Stage 5: Rounding (RNE - Round to Nearest Even)
    // =========================================================
    // FP9 目标：E5M3。我们需要保留 3 位尾数。
    // full_mant 结构: [10: Integer/Hidden] . [9 8 7: FP9 Mant] [6: Guard] [5..0: Sticky]
    
    // 提取 FP9 候选尾数 (3 bits)
    // 如果是 Normal (true_exp > 0)，bit 10 是隐藏位，存储 bits 9-7
    // 如果是 Subnorm (true_exp == 0)，bit 10 是 0，存储 bits 9-7
    int fp9_mant = (full_mant >> 7) & 7;
    
    // 舍入位分析
    bool guard = (full_mant >> 6) & 1;
    bool round_sticky = (full_mant & 0x3F) != 0; // Bits 5-0
    
    // RNE 逻辑:
    // Round up if: (Guard == 1) AND (Sticky != 0 OR LSB == 1)
    if (guard && (round_sticky || (fp9_mant & 1))) {
        fp9_mant++;
        // 处理尾数进位 (3 bits overflow -> 1000)
        if (fp9_mant > 7) {
            fp9_mant = 0;
            true_exp++;
        }
    }

    // =========================================================
    // Stage 6: Final Packing & Overflow Check
    // =========================================================
    
    // 检查舍入后是否溢出 (Exp >= 31)
    if (true_exp >= 31) {
        return (sign << 8) | (0x1F << 3); // Inf
    }

    // 组装: Sign | Exp | Mant
    return (sign << 8) | (true_exp << 3) | fp9_mant;
}