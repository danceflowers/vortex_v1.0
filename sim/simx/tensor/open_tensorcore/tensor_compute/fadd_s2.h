#pragma once
#include "fadd_s1.h"
// =============================================================================
// fadd_s2: Rounding and result assembly
// Matches fadd_s2.v exactly
// =============================================================================


inline uint32_t fadd_s2(const fadd_s1_out& s1, int EXPWIDTH, int PRECISION)
{
    const int NEAR_INV = (1 << EXPWIDTH) - 2;
    const int INV      = (1 << EXPWIDTH) - 1;
    RoundingMode rm = (RoundingMode)s1.rm;

    // Special output
    if (s1.special_case_valid) {
        if (s1.special_case_nan) {
            // NaN: {0, all-ones exp, 1, zeros}
            uint32_t nan_sig = 1u << (PRECISION - 2);
            return (0u << (EXPWIDTH + PRECISION - 1)) |
                   ((uint32_t)INV << (PRECISION - 1)) | nan_sig;
        }
        // Inf
        return (0u << (EXPWIDTH + PRECISION - 1)) |
               ((uint32_t)INV << (PRECISION - 1));
    }

    // ── Far path rounding ──
    // rounder_1_in = far_sig[PRECISION+1:0]
    uint32_t far_r1_in = s1.far_sig & ((1u << (PRECISION + 2)) - 1);
    uint32_t far_r1_data = (far_r1_in >> 3) & ((1u << (PRECISION - 1)) - 1);
    bool far_r1_round  = (far_r1_in >> 2) & 1;
    bool far_r1_sticky = (far_r1_in & 3) != 0;
    RoundResult far_rr = do_rounding(far_r1_data, PRECISION - 1, s1.far_sign, far_r1_round, far_r1_sticky, rm);

    int far_exp_rounded = (int)far_rr.cout + s1.far_exp;
    bool far_of_before = (s1.far_exp == INV);
    bool far_of_after  = far_rr.cout && (s1.far_exp == NEAR_INV);
    bool far_of = far_of_before || far_of_after || s1.far_mul_of;
    bool far_ix = far_rr.inexact || far_of;

    uint32_t far_result = ((uint32_t)s1.far_sign << (EXPWIDTH + PRECISION - 1)) |
                          ((far_exp_rounded & ((1 << EXPWIDTH) - 1)) << (PRECISION - 1)) |
                          (far_rr.out & ((1u << (PRECISION - 1)) - 1));

    // ── Near path rounding ──
    bool near_is_zero = (s1.near_exp == 0) && s1.near_sig_is_zero;

    uint32_t near_r1_in = s1.near_sig & ((1u << (PRECISION + 2)) - 1);
    uint32_t near_r1_data = (near_r1_in >> 3) & ((1u << (PRECISION - 1)) - 1);
    bool near_r1_round  = (near_r1_in >> 2) & 1;
    bool near_r1_sticky = (near_r1_in & 3) != 0;
    RoundResult near_rr = do_rounding(near_r1_data, PRECISION - 1, s1.near_sign, near_r1_round, near_r1_sticky, rm);

    int near_exp_rounded = (int)near_rr.cout + s1.near_exp;
    bool near_zero_sign = (rm == RDN);
    bool near_sign_out  = (s1.near_sign && !near_is_zero) || (near_zero_sign && near_is_zero);
    bool near_of = (near_exp_rounded == ((1 << EXPWIDTH) - 1));
    bool near_ix = near_rr.inexact || near_of;

    uint32_t near_result = ((uint32_t)near_sign_out << (EXPWIDTH + PRECISION - 1)) |
                           ((near_exp_rounded & ((1 << EXPWIDTH) - 1)) << (PRECISION - 1)) |
                           (near_rr.out & ((1u << (PRECISION - 1)) - 1));

    // Common overflow handling
    bool common_of = s1.sel_far_path ? far_of : near_of;
    if (common_of) {
        bool of_sign = s1.sel_far_path ? s1.far_sign : s1.near_sign;
        bool rmin = (rm == RTZ) || (rm == RDN && !of_sign) || (rm == RUP && of_sign);
        int of_exp = rmin ? NEAR_INV : INV;
        int of_sig = rmin ? ((1 << (PRECISION - 1)) - 1) : 0;
        return ((uint32_t)of_sign << (EXPWIDTH + PRECISION - 1)) |
               ((of_exp & ((1 << EXPWIDTH) - 1)) << (PRECISION - 1)) |
               (of_sig & ((1u << (PRECISION - 1)) - 1));
    }

    return s1.sel_far_path ? far_result : near_result;
}