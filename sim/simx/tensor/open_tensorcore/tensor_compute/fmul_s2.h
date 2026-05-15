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
    (void)a_bits;
    (void)b_bits;
    (void)EXPWIDTH;
    (void)PRECISION;
    fmul_s2_out out;
    out.s1 = s1;
    // Stage 1 already unpacked FP9 into the internal significands consumed by
    // the FP22-domain multiplier array.
    out.prod = s1.raw_a_sig * s1.raw_b_sig;
    return out;
}
