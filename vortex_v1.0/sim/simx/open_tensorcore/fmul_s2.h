#pragma once
#include "fmul_s1.h"
// =============================================================================
// fmul_s2: Mantissa multiplication (matches fmul_s2.v — passthrough + product)
// The actual multiplication is done by naivemultiplier between s1 and s2 registers
// =============================================================================
//#include "fp_arith.h"

struct fmul_s2_out {
    uint32_t prod;  // 2*PRECISION bits
    // All s1 fields passed through
    fmul_s1_out s1;
};

inline fmul_s2_out fmul_s2(uint32_t a_bits, uint32_t b_bits, int EXPWIDTH, int PRECISION,
                          const fmul_s1_out& s1)
{
    fmul_s2_out out;
    out.s1 = s1;

    // Extract significands with hidden bit (same as s1)
    uint32_t a_exp_raw = (a_bits >> (PRECISION - 1)) & ((1u << EXPWIDTH) - 1);
    uint32_t b_exp_raw = (b_bits >> (PRECISION - 1)) & ((1u << EXPWIDTH) - 1);
    uint32_t a_mant = a_bits & ((1u << (PRECISION - 1)) - 1);
    uint32_t b_mant = b_bits & ((1u << (PRECISION - 1)) - 1);
    bool a_exp_is_zero = (a_exp_raw == 0);
    bool b_exp_is_zero = (b_exp_raw == 0);

    uint32_t raw_a_sig = (a_exp_is_zero ? 0 : (1u << (PRECISION - 1))) | a_mant;
    uint32_t raw_b_sig = (b_exp_is_zero ? 0 : (1u << (PRECISION - 1))) | b_mant;

    // Naive multiplier: PRECISION × PRECISION = 2*PRECISION bits
    out.prod = raw_a_sig * raw_b_sig;
    return out;
}