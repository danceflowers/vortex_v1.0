#pragma once
#include <array>
#include <cstdlib>
#include <cstdint>
#include "fp_types.h"
#include "tc_mul_pipe.h"
#include "tc_add_pipe.h"
#include "tc_add4_pipe_fp22.h"
#include "tc_select_pipe.h"

// ============================================================================
// tc_mul_add -- one scalar dot-product lane in the 8x8 TensorCore array.
// ============================================================================
//
// This lane computes one output element:
//   D[i][j] = sum_k(A[i][k] * B[k][j]) + C[i][j]
//
// Pipeline:
//   - select_pipe picks four products from the 8-wide source operands.
//   - 4 parallel mul_pipe instances produce FP22 products.
//   - add4_pipe_fp22 reduces the four products.
//   - final_add adds the C value carried through the pipeline as passthrough.
// ============================================================================

struct tc_mul_add {

    // Arithmetic sub-pipelines.

    tc_select_pipe select_pipe;
    std::array<mul_pipe, 4> mul_array;
    add4_pipe_fp22 add4;
    add_pipe_fp22 final_add;

    // Input latch written by TensorCoreTop before tick().
    struct {
        uint32_t a_in[8];       // Source A[i][k], k = 0..7, in FP9 format.
        uint32_t b_in[8];       // Source B[k][j], k = 0..7, in FP9 format.
        uint32_t c_in;          // Passthrough C value for non-resident mode.
        uint32_t sparsity_kind;
        uint16_t sparse_row_meta;
        bool input_valid;
        TensorCoreMeta meta;
    } mul_add_input;

    // Final output register.
    struct {
        uint32_t fp22_result;
        bool valid;
        TensorCoreMeta meta;
    } r2;

    void reset() {
        select_pipe.reset();
        for (int i = 0; i < 4; i++) mul_array[i].reset();
        add4.reset();
        final_add.reset();
        r2.fp22_result = 0;
        r2.valid = false;
        r2.meta = {};
        mul_add_input.meta = {};
        mul_add_input.sparsity_kind = vortex::tensor::sparse_none;
        mul_add_input.sparse_row_meta = 0;
        mul_add_input.input_valid = false;
    }

    // Public status/query interface.

    bool out_valid() const { return r2.valid; }

    bool active() const {
        if (mul_add_input.input_valid || r2.valid) return true;
        if (select_pipe.active()) return true;
        for (const auto& mul : mul_array) { if (mul.active()) return true; }
        if (add4.active() || final_add.active()) return true;
        return false;
    }

    bool in_ready(bool out_ready) const {
        bool s6_ready = out_ready || !r2.valid;
        bool s5_ready = final_add.in_ready(s6_ready);
        bool s4_ready = add4.in_ready(s5_ready);
        bool s3_ready = true;
        for (int i = 0; i < 4; ++i) s3_ready &= mul_array[i].in_ready(s4_ready);
        return select_pipe.in_ready(s3_ready);
    }

    const uint32_t& out_fp22() const { return r2.fp22_result; }
    const TensorCoreMeta& out_meta() const { return r2.meta; }

    // Advance the full scalar lane by one cycle.
    void tick(bool out_ready, const Config& g_cfg) {
        // Ready chain from output register back to multiplier inputs.
        bool s6_ready = out_ready || !r2.valid;
        bool s5_ready = final_add.in_ready(s6_ready);
        bool s4_ready = add4.in_ready(s5_ready);
        bool s3_ready = true;
        for (int i = 0; i < 4; ++i) s3_ready &= mul_array[i].in_ready(s4_ready);
        bool s2_ready = select_pipe.in_ready(s3_ready);

        // ======== s6: final_add → r2 ========
        if (s6_ready) {
            if (final_add.out_valid()) {
                r2.fp22_result = final_add.out_data();
                r2.valid = true;
                r2.meta = final_add.out_meta();
            } else {
                r2.valid = false;
                r2.meta = {};
            }
        }

        // ======== s5: add4 → final_add ========
        if (s5_ready) {
            if (add4.out_valid()) {
                final_add.add_input.a_in = add4.out_data();
                final_add.add_input.b_in = add4.out_passthrough();
                final_add.add_input.input_valid = true;
                final_add.add_input.meta = add4.out_meta();
            } else {
                final_add.add_input.input_valid = false;
                final_add.add_input.meta = {};
            }
            final_add.tick(s6_ready, g_cfg);
        }

        // ======== s4: mul → add4 ========
        bool r3_valid = true;
        for (int i = 0; i < 4; ++i) r3_valid &= mul_array[i].out_valid();

        if (s4_ready) {
            if (r3_valid) {
                for (int i = 0; i < 4; ++i) {
                    add4.add_input.in[i] = mul_array[i].out_data();
                }
                add4.add_input.input_valid = true;
                add4.add_input.meta = mul_array[0].out_meta();
            } else {
                add4.add_input.input_valid = false;
                add4.add_input.meta = {};
            }
            uint32_t c_in = r3_valid ? mul_array[0].r3.c_in : 0;
            add4.tick(s5_ready, g_cfg, c_in);
        }

        // ======== s3: select → mul ========
        if (s3_ready) {
            uint32_t c_in = select_pipe.out_valid() ? select_pipe.out_c() : 0;
            if (select_pipe.out_valid()) {
                for (int i = 0; i < 4; ++i) {
                    mul_array[i].mul_input.a_fp9 = select_pipe.out_a(i);
                    mul_array[i].mul_input.b_fp9 = select_pipe.out_b(i);
                    mul_array[i].mul_input.input_valid = true;
                    mul_array[i].mul_input.meta = select_pipe.out_meta();
                    mul_array[i].tick(s4_ready, g_cfg, c_in);
                }
            } else {
                for (int i = 0; i < 4; ++i) {
                    mul_array[i].mul_input.input_valid = false;
                    mul_array[i].mul_input.meta = {};
                    mul_array[i].tick(s4_ready, g_cfg, c_in);
                }
            }
        }

        // ======== s2: external input → select ========
        if (s2_ready) {
            if (mul_add_input.input_valid) {
                for (int i = 0; i < 8; ++i) {
                    select_pipe.select_input.a_src[i] = mul_add_input.a_in[i];
                    select_pipe.select_input.b_src[i] = mul_add_input.b_in[i];
                }
                select_pipe.select_input.c_in = mul_add_input.c_in;
                select_pipe.select_input.sparsity_kind = mul_add_input.sparsity_kind;
                select_pipe.select_input.sparse_row_meta = mul_add_input.sparse_row_meta;
                select_pipe.select_input.input_valid = true;
                select_pipe.select_input.meta = mul_add_input.meta;
            } else {
                select_pipe.select_input.input_valid = false;
                select_pipe.select_input.meta = {};
                select_pipe.select_input.sparsity_kind = vortex::tensor::sparse_none;
                select_pipe.select_input.sparse_row_meta = 0;
            }
            select_pipe.tick(s3_ready);
        }
    }
};
