#pragma once
#include "fmul_s2.h"
#include "fp_types.h"
inline uint32_t fmul_s3(const fmul_s2_out& s2, int EXPWIDTH, int PRECISION)
{
    const int PADDINGBITS = PRECISION + 2;
    const int NEAR_INV    = (1 << EXPWIDTH) - 2;
    const int INV         = (1 << EXPWIDTH) - 1;
    RoundingMode rm = (RoundingMode)s2.s1.rm;

    // sig_shifter_in = {PADDINGBITS zeros, product}
    // Total width = PRECISION*3+2
    int total_width = PRECISION * 3 + 2;
    uint64_t sig_shifter_in = (uint64_t)s2.prod; // product in low bits, high bits 0
    uint64_t sig_shifted_long = sig_shifter_in << s2.s1.shift_amt;
    uint64_t sig_shifted_raw  = sig_shifted_long & ((1ULL << total_width) - 1);

    bool exp_is_subnormal = s2.s1.may_be_subnormal && !((sig_shifted_raw >> (total_width - 1)) & 1);
    bool no_extra_shift   = ((sig_shifted_raw >> (total_width - 1)) & 1) || exp_is_subnormal;

    int exp_pre_round;
    if (exp_is_subnormal)
        exp_pre_round = 0;
    else if (no_extra_shift)
        exp_pre_round = s2.s1.exp_shifted;
    else
        exp_pre_round = s2.s1.exp_shifted - 1;

    uint64_t sig_shifted;
    if (no_extra_shift)
        sig_shifted = sig_shifted_raw;
    else
        sig_shifted = ((sig_shifted_raw & ((1ULL << (total_width - 1)) - 1)) << 1);

    // Extract raw_in fields
    bool raw_in_sign = s2.s1.prod_sign;
    int  raw_in_exp  = exp_pre_round & ((1 << EXPWIDTH) - 1);

    // raw_in_sig = {sig_shifted[top PRECISION+2 bits], | sig_shifted[PRECISION+1:0]}
    // Width of raw_in_sig = PRECISION + 3
    uint32_t top_bits = (sig_shifted >> (PRECISION * 2)) & ((1u << (PRECISION + 2)) - 1);
    bool sticky_low   = (sig_shifted & ((1ULL << (PRECISION + 2)) - 1)) != 0;
    uint32_t raw_in_sig = (top_bits << 1) | (sticky_low ? 1 : 0); // PRECISION+3 bits


//###############

    // Rounder 1 input: raw_in_sig[PRECISION+1:0]
    uint32_t rounder1_in = raw_in_sig & ((1u << (PRECISION + 2)) - 1);
    // in[PRECISION+1:3], roundin = [2], stickyin = |[1:0]
    uint32_t r1_data = (rounder1_in >> 3) & ((1u << (PRECISION - 1)) - 1);
    bool r1_roundin  = (rounder1_in >> 2) & 1;
    bool r1_stickyin = (rounder1_in & 0x3) != 0;
    RoundResult rr1 = do_rounding(r1_data, PRECISION - 1, raw_in_sign, r1_roundin, r1_stickyin, rm);

    // Common case
    int exp_rounded = (int)rr1.cout + raw_in_exp;
    bool common_of  = (rr1.cout ? (raw_in_exp == NEAR_INV) : (raw_in_exp == INV)) || s2.s1.early_overflow;
    bool common_ix  = rr1.inexact | common_of;
    // Tininess check
    uint32_t top2 = (raw_in_sig >> (PRECISION + 1)) & 3;
    bool tininess = (top2 == 0) || (top2 == 1 && !rr1.cout);
    // Rounder 0 for tininess (optional, matching RTL)
    // bool common_uf = tininess & common_ix; // not needed for result

    bool rmin = (rm == RTZ) || (rm == RDN && !raw_in_sign) || (rm == RUP && raw_in_sign);
    int of_exp   = rmin ? NEAR_INV : INV;
    int com_exp  = common_of ? of_exp : exp_rounded;
    int com_sig  = common_of ? (rmin ? ((1 << (PRECISION - 1)) - 1) : 0) : (int)rr1.out;

    uint32_t common_result = ((uint32_t)raw_in_sign << (EXPWIDTH + PRECISION - 1)) |
                             ((com_exp & ((1 << EXPWIDTH) - 1)) << (PRECISION - 1)) |
                             (com_sig & ((1 << (PRECISION - 1)) - 1));

    // Special cases
    if (s2.s1.special_case_valid) {
        int sp_exp = s2.s1.special_case_inf ? INV : 0;
        int sp_sig = 0;
        if (s2.s1.special_case_nan) {
            // QNaN: exp=all ones, mantissa MSB=1
            sp_exp = INV;
            sp_sig = 1 << (PRECISION - 2); // set quiet bit
        }
        return ((uint32_t)raw_in_sign << (EXPWIDTH + PRECISION - 1)) |
               ((sp_exp & ((1 << EXPWIDTH) - 1)) << (PRECISION - 1)) |
               (sp_sig & ((1 << (PRECISION - 1)) - 1));
    }

    return common_result;
}