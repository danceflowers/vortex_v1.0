#pragma once
#include "fp_types.h"
// #include "fp_arith.h"
// =============================================================================
// fmul_s1: Exponent calculation, special case detection
// =============================================================================
struct fmul_s1_out {
    bool special_case_valid;
    bool special_case_nan;
    bool special_case_inf;
    bool special_case_inv;
    bool special_case_haszero;
    bool early_overflow;
    bool prod_sign;
    int  shift_amt;
    int  exp_shifted;
    bool may_be_subnormal;
    int  rm;
    uint32_t raw_a_sig;
    uint32_t raw_b_sig;
};


inline fmul_s1_out fmul_s1(uint32_t a_bits, uint32_t b_bits, int EXPWIDTH, int PRECISION, RoundingMode rm)
{
    fmul_s1_out out = {};
    constexpr int kFp9ExpWidth = 5;
    constexpr int kFp9MantWidth = 3;
    constexpr int kFp9Bias = 15;
    const int PADDINGBITS = PRECISION + 2;
    const int BIASINT     = (1 << (EXPWIDTH - 1)) - 1;
    const int MAXNORMEXP  = (1 << EXPWIDTH) - 2;

    // Inputs are raw FP9 (E5M3). Stage 1 widens their field interpretation
    // into the internal FP22 arithmetic domain rather than relying on an
    // explicit external fp9->fp22 conversion stage.
    bool a_sign = (a_bits >> (kFp9ExpWidth + kFp9MantWidth)) & 1;
    bool b_sign = (b_bits >> (kFp9ExpWidth + kFp9MantWidth)) & 1;
    uint32_t a_exp_fp9 = (a_bits >> kFp9MantWidth) & ((1u << kFp9ExpWidth) - 1);
    uint32_t b_exp_fp9 = (b_bits >> kFp9MantWidth) & ((1u << kFp9ExpWidth) - 1);
    uint32_t a_mant_fp9 = a_bits & ((1u << kFp9MantWidth) - 1);
    uint32_t b_mant_fp9 = b_bits & ((1u << kFp9MantWidth) - 1);

    uint32_t a_exp_raw = 0;
    uint32_t b_exp_raw = 0;
    uint32_t a_mant = 0;
    uint32_t b_mant = 0;

    if (a_exp_fp9 == ((1u << kFp9ExpWidth) - 1)) {
        a_exp_raw = (1u << EXPWIDTH) - 1;
        a_mant = a_mant_fp9 ? ((1u << (PRECISION - 2)) | (a_mant_fp9 << (PRECISION - 4))) : 0;
    } else if (a_exp_fp9 == 0 && a_mant_fp9 == 0) {
        a_exp_raw = 0;
        a_mant = 0;
    } else if (a_exp_fp9 == 0) {
        int lz = clz(a_mant_fp9, kFp9MantWidth);
        a_exp_raw = BIASINT - kFp9Bias + 1 - lz;
        a_mant = (((a_mant_fp9 << (1 + lz)) & ((1u << kFp9MantWidth) - 1)) << (PRECISION - 4)) &
                 ((1u << (PRECISION - 1)) - 1);
    } else {
        a_exp_raw = a_exp_fp9 + (BIASINT - kFp9Bias);
        a_mant = (a_mant_fp9 << (PRECISION - 4)) & ((1u << (PRECISION - 1)) - 1);
    }

    if (b_exp_fp9 == ((1u << kFp9ExpWidth) - 1)) {
        b_exp_raw = (1u << EXPWIDTH) - 1;
        b_mant = b_mant_fp9 ? ((1u << (PRECISION - 2)) | (b_mant_fp9 << (PRECISION - 4))) : 0;
    } else if (b_exp_fp9 == 0 && b_mant_fp9 == 0) {
        b_exp_raw = 0;
        b_mant = 0;
    } else if (b_exp_fp9 == 0) {
        int lz = clz(b_mant_fp9, kFp9MantWidth);
        b_exp_raw = BIASINT - kFp9Bias + 1 - lz;
        b_mant = (((b_mant_fp9 << (1 + lz)) & ((1u << kFp9MantWidth) - 1)) << (PRECISION - 4)) &
                 ((1u << (PRECISION - 1)) - 1);
    } else {
        b_exp_raw = b_exp_fp9 + (BIASINT - kFp9Bias);
        b_mant = (b_mant_fp9 << (PRECISION - 4)) & ((1u << (PRECISION - 1)) - 1);
    }

    bool a_exp_is_zero = (a_exp_raw == 0);
    bool b_exp_is_zero = (b_exp_raw == 0);
    bool a_exp_is_ones = (a_exp_raw == ((1u << EXPWIDTH) - 1));
    bool b_exp_is_ones = (b_exp_raw == ((1u << EXPWIDTH) - 1));
    bool a_sig_is_zero = (a_mant == 0);
    bool b_sig_is_zero = (b_mant == 0);

    bool a_is_inf  = a_exp_is_ones && a_sig_is_zero;
    bool b_is_inf  = b_exp_is_ones && b_sig_is_zero;
    bool a_is_zero = a_exp_is_zero && a_sig_is_zero;
    bool b_is_zero = b_exp_is_zero && b_sig_is_zero;
    bool a_is_nan  = a_exp_is_ones && !a_sig_is_zero;
    bool b_is_nan  = b_exp_is_ones && !b_sig_is_zero;
    bool a_is_snan = a_is_nan && !((a_mant >> (PRECISION - 2)) & 1);
    bool b_is_snan = b_is_nan && !((b_mant >> (PRECISION - 2)) & 1);

    // raw_exp = exp | {0..0, exp_is_zero} (force subnormal exp to 1)
    int raw_a_exp = a_exp_raw | (a_exp_is_zero ? 1 : 0);
    int raw_b_exp = b_exp_raw | (b_exp_is_zero ? 1 : 0);
    // Significand with hidden bit
    int raw_a_sig = (a_exp_is_zero ? 0 : (1 << (PRECISION - 1))) | a_mant;
    int raw_b_sig = (b_exp_is_zero ? 0 : (1 << (PRECISION - 1))) | b_mant;
    out.raw_a_sig = raw_a_sig;
    out.raw_b_sig = raw_b_sig;

    out.prod_sign = a_sign ^ b_sign;

    // Exponent calculation
    int exp_sum       = raw_a_exp + raw_b_exp;
    int prod_exp      = exp_sum - (BIASINT - (PADDINGBITS + 1));
    int shift_lim_sub = exp_sum - (BIASINT - PADDINGBITS);
    bool prod_exp_uf  = (shift_lim_sub < 0);
    int shift_lim     = prod_exp_uf ? 0 : shift_lim_sub;
    bool prod_exp_ov  = (exp_sum > (MAXNORMEXP + BIASINT));

    // Subnormal shift calculation
    int subnormal_sig = a_exp_is_zero ? raw_a_sig : raw_b_sig;
    int lzc_width = PRECISION * 2 + 2;
    int lzc_val = clz(subnormal_sig, lzc_width);

    bool exceed_lim = (shift_lim <= lzc_val);
    int shift_amt   = prod_exp_uf ? 0 : (exceed_lim ? shift_lim : lzc_val);
    int exp_shifted  = prod_exp - shift_amt;

    out.early_overflow   = prod_exp_ov;
    out.shift_amt        = shift_amt;
    out.exp_shifted      = exp_shifted;
    out.may_be_subnormal = exceed_lim || prod_exp_uf;
    out.rm               = rm;

    // Special cases
    bool has_zero = a_is_zero || b_is_zero;
    bool has_nan  = a_is_nan  || b_is_nan;
    bool has_snan = a_is_snan || b_is_snan;
    bool has_inf  = a_is_inf  || b_is_inf;
    bool zero_mul_inf = has_zero && has_inf;

    out.special_case_valid   = has_zero || has_nan || has_inf;
    out.special_case_nan     = has_nan || zero_mul_inf;
    out.special_case_inf     = has_inf; // Note: RTL assigns has_inf, not has_inf && !zero_mul_inf here
    out.special_case_inv     = has_snan || zero_mul_inf;
    out.special_case_haszero = has_zero;

    return out;
}
