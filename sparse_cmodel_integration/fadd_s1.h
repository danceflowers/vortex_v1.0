

#include "fp_types.h"
#pragma once
// =============================================================================
// far_path: Addition path for |exp_diff| > 1 or effective addition
// =============================================================================

struct FarPathOut {
    bool     result_sign;
    uint32_t result_exp;
    uint32_t result_sig;
};

inline FarPathOut far_path_compute(bool a_sign, int a_exp, uint32_t a_sig,
                                    uint32_t b_sig, int expdiff, bool effsub,
                                    bool small_add, int EXPWIDTH, int PRECISION, int OUTPC)
{
    FarPathOut out;
    // Shift B's significand right by expdiff
    uint32_t b_shifted = 0;
    bool sticky = false;

    if (expdiff < (int)(PRECISION + 3)) {
        uint32_t mask = (1u << expdiff) - 1;
        sticky = (b_sig & mask) != 0;
        b_shifted = b_sig >> expdiff;
    } else {
        sticky = (b_sig != 0);
        b_shifted = 0;
    }

    // Addition/subtraction
    int sig_result;
    if (effsub) {
        sig_result = (int)a_sig - (int)b_shifted;
    } else {
        sig_result = (int)a_sig + (int)b_shifted;
        // Check carry: bit[PRECISION]
        if ((sig_result >> PRECISION) & 1) {
            sticky = sticky || (sig_result & 1);
            sig_result >>= 1;
            a_exp += 1;
        }
    }

    if (small_add) {
        out.result_exp = 0;
    } else {
        out.result_exp = a_exp;
    }

    out.result_sign = a_sign;
    // sig has OUTPC+3 bits: {sig_result[OUTPC+1:0], sticky}
    // But RTL stores (OUTPC+3) bits total
    // Construct output sig: take top (OUTPC+2) bits of sig_result, append sticky
    // We need to right-align to OUTPC+3 bits
    int shift = PRECISION - OUTPC - 2;
    uint32_t top_sig;
    bool extra_sticky;
    if (shift > 0) {
        extra_sticky = (sig_result & ((1 << shift) - 1)) != 0;
        top_sig = sig_result >> shift;
    } else {
        extra_sticky = false;
        top_sig = sig_result << (-shift);
    }
    out.result_sig = ((top_sig & ((1u << (OUTPC + 2)) - 1)) << 1) | (sticky || extra_sticky ? 1 : 0);

    return out;
}

// =============================================================================
// near_path: Subtraction path for |exp_diff| ≤ 1
// =============================================================================
struct NearPathOut {
    bool     result_sign;
    uint32_t result_exp;
    uint32_t result_sig;
    bool     sig_is_zero;
    bool     a_lt_b;
};

inline NearPathOut near_path_compute(bool a_sign, int a_exp, uint32_t a_sig,
                                      bool b_sign, uint32_t b_sig, bool need_shift_b,
                                      int EXPWIDTH, int PRECISION, int OUTPC)
{
    NearPathOut out;

    uint32_t b_sig_aligned = need_shift_b ? (b_sig >> 1) : b_sig;

    bool a_lt_b = (a_sig < b_sig_aligned);
    int sig_diff;
    if (a_lt_b) {
        sig_diff = (int)b_sig_aligned - (int)a_sig;
        out.result_sign = b_sign;
    } else {
        sig_diff = (int)a_sig - (int)b_sig_aligned;
        out.result_sign = a_sign;
    }

    out.sig_is_zero = (sig_diff == 0);
    out.a_lt_b = a_lt_b;

    // Normalize: count leading zeros and shift
    int lzc_val = clz(sig_diff, PRECISION + 1);
    uint32_t sig_normalized = (uint32_t)sig_diff << lzc_val;
    int exp_normalized = a_exp - lzc_val;
    if (exp_normalized <= 0) exp_normalized = 0;

    out.result_exp = exp_normalized;
    // Map to OUTPC+3 bit output
    int shift = PRECISION - OUTPC - 2;
    if (shift > 0)
        out.result_sig = sig_normalized >> shift;
    else
        out.result_sig = sig_normalized << (-shift);
    out.result_sig &= ((1u << (OUTPC + 3)) - 1);

    return out;
}

// =============================================================================
// fadd_s1: Path selection + parallel near/far computation
// Matches fadd_s1.v exactly
// =============================================================================
struct fadd_s1_out {
    int  rm;
    bool far_sign;
    int  far_exp;
    uint32_t far_sig;
    bool near_sign;
    int  near_exp;
    uint32_t near_sig;
    bool special_case_valid;
    bool special_case_iv;
    bool special_case_nan;
    bool special_case_inf_sign;
    bool small_add;
    bool far_mul_of;
    bool near_sig_is_zero;
    bool sel_far_path;
};

inline fadd_s1_out fadd_s1(uint32_t a_bits, uint32_t b_bits, int EXPWIDTH, int PRECISION, int OUTPC, RoundingMode rm)
{
    fadd_s1_out out = {};

    // Classify
    uint32_t a_exp_raw = (a_bits >> (PRECISION - 1)) & ((1u << EXPWIDTH) - 1);
    uint32_t b_exp_raw = (b_bits >> (PRECISION - 1)) & ((1u << EXPWIDTH) - 1);
    uint32_t a_mant = a_bits & ((1u << (PRECISION - 1)) - 1);
    uint32_t b_mant = b_bits & ((1u << (PRECISION - 1)) - 1);
    bool a_sign = (a_bits >> (EXPWIDTH + PRECISION - 1)) & 1;
    bool b_sign = (b_bits >> (EXPWIDTH + PRECISION - 1)) & 1;

    bool a_exp_is_zero = (a_exp_raw == 0);
    bool b_exp_is_zero = (b_exp_raw == 0);
    bool a_exp_is_ones = (a_exp_raw == ((1u << EXPWIDTH) - 1));
    bool b_exp_is_ones = (b_exp_raw == ((1u << EXPWIDTH) - 1));
    bool a_sig_is_zero = (a_mant == 0);
    bool b_sig_is_zero = (b_mant == 0);

    bool a_is_inf  = a_exp_is_ones && a_sig_is_zero;
    bool b_is_inf  = b_exp_is_ones && b_sig_is_zero;
    bool a_is_nan  = a_exp_is_ones && !a_sig_is_zero;
    bool b_is_nan  = b_exp_is_ones && !b_sig_is_zero;
    bool a_is_snan = a_is_nan && !((a_mant >> (PRECISION - 2)) & 1);
    bool b_is_snan = b_is_nan && !((b_mant >> (PRECISION - 2)) & 1);

    // Raw fields (force subnormal exp to 1)
    int raw_a_exp = a_exp_raw | (a_exp_is_zero ? 1 : 0);
    int raw_b_exp = b_exp_raw | (b_exp_is_zero ? 1 : 0);
    uint32_t raw_a_sig = (a_exp_is_zero ? 0 : (1u << (PRECISION - 1))) | a_mant;
    uint32_t raw_b_sig = (b_exp_is_zero ? 0 : (1u << (PRECISION - 1))) | b_mant;

    bool eff_sub   = a_sign ^ b_sign;
    bool small_add = a_exp_is_zero && b_exp_is_zero;

    // Special cases
    bool special_has_nan  = a_is_nan || b_is_nan;
    bool special_has_snan = a_is_snan || b_is_snan;
    bool special_has_inf  = a_is_inf || b_is_inf;
    bool inf_iv = a_is_inf && b_is_inf && eff_sub;
    out.special_case_valid = special_has_nan || special_has_inf;
    out.special_case_iv    = special_has_snan || inf_iv;
    out.special_case_nan   = special_has_nan || inf_iv;
    out.special_case_inf_sign = a_is_inf ? a_sign : b_sign;
    out.small_add = small_add;
    out.far_mul_of = b_exp_is_ones && !eff_sub;

    // Path selection
    int exp_diff_a_b = raw_a_exp - raw_b_exp;
    int exp_diff_b_a = raw_b_exp - raw_a_exp;
    bool need_swap = (exp_diff_a_b < 0);
    int ea_minus_eb = need_swap ? exp_diff_b_a : exp_diff_a_b;
    out.sel_far_path = !eff_sub || (ea_minus_eb > 1);

    // Far path
    bool far_a_sign    = need_swap ? b_sign : a_sign;
    int  far_a_exp     = need_swap ? raw_b_exp : raw_a_exp;
    uint32_t far_a_sig = need_swap ? raw_b_sig : raw_a_sig;
    uint32_t far_b_sig = need_swap ? raw_a_sig : raw_b_sig;

    FarPathOut fpo = far_path_compute(far_a_sign, far_a_exp, far_a_sig,
                                       far_b_sig, ea_minus_eb, eff_sub, small_add,
                                       EXPWIDTH, PRECISION, OUTPC);
    out.far_sign = fpo.result_sign;
    out.far_exp  = fpo.result_exp;
    out.far_sig  = fpo.result_sig;

    // Near path (two instances, select based on swap)
    bool near_exp_neq = (raw_a_exp != raw_b_exp);

    NearPathOut np0 = near_path_compute(a_sign, raw_a_exp, raw_a_sig,
                                         b_sign, raw_b_sig, near_exp_neq,
                                         EXPWIDTH, PRECISION, OUTPC);
    NearPathOut np1 = near_path_compute(b_sign, raw_b_exp, raw_b_sig,
                                         a_sign, raw_a_sig, near_exp_neq,
                                         EXPWIDTH, PRECISION, OUTPC);

    bool near_sel = need_swap || (!near_exp_neq && np0.a_lt_b);
    out.near_sign         = near_sel ? np1.result_sign : np0.result_sign;
    out.near_exp          = near_sel ? (int)np1.result_exp : (int)np0.result_exp;
    out.near_sig          = near_sel ? np1.result_sig : np0.result_sig;
    out.near_sig_is_zero  = near_sel ? np1.sig_is_zero : np0.sig_is_zero;
    out.rm = rm;

    return out;
}
