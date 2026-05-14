#pragma once
#include <cstdint>
#include "config_register.h"
#include "fmul_s1.h"
#include "fmul_s2.h"
#include "fmul_s3.h"

// Three-stage FP9 multiplier pipeline used by each dot-product lane.
//
// Stage 1 expands raw FP9 inputs into the internal FP22 arithmetic domain and
// detects special cases. Stage 2 multiplies the significands. Stage 3
// normalizes, rounds, and packs the result. The valid/ready chain propagates
// backpressure from r3 to r1 so stalled downstream consumers do not overwrite
// in-flight products.

struct mul_pipe {

    // Input latch written by the owner before tick().
    struct {
        uint32_t a_fp9;           // FP9 operand A (low 9 bits are used).
        uint32_t b_fp9;           // FP9 operand B (low 9 bits are used).
        bool input_valid;         // Input valid bit.
        TensorCoreMeta meta;      // Metadata carried with the product.
    } mul_input;

    // Pipeline registers.

    // Stage 1 register: expanded operands and special-case flags.
    struct {
        fmul_s1_out s1_result;    // Exponent and special-case fields.
        uint16_t a_fp9;           // Original FP9 operand A.
        uint16_t b_fp9;           // Original FP9 operand B.
        uint32_t c_in;            // Passthrough accumulator input.
        bool valid;               // Stage valid bit.
        TensorCoreMeta meta;      // Metadata carried with the product.
    } r1;

    // Stage 2 register: raw product plus stage-1 passthrough fields.
    struct {
        fmul_s2_out s2_result;    // Product and forwarded s1 state.
        uint32_t c_in;            // Passthrough accumulator input.
        bool valid;               // Stage valid bit.
        TensorCoreMeta meta;      // Metadata carried with the product.
    } r2;

    // Stage 3 register: normalized and rounded packed product.
    struct {
        uint32_t result;          // Packed floating-point product.
        uint32_t c_in;            // Passthrough accumulator input.
        bool valid;               // Output valid bit.
        TensorCoreMeta meta;      // Metadata carried with the product.
    } r3;

    // Clear all pipeline valid bits.
    void reset() {
        r1.valid = false;
        r2.valid = false;
        r3.valid = false;
        mul_input.input_valid = false;
        mul_input.meta = {};
    }

    // External query interface.

    // True when r3 holds a completed product.
    bool out_valid() const {
        return r3.valid;
    }

    // True while any stage holds an in-flight product.
    bool active() const {
        return mul_input.input_valid || r1.valid || r2.valid || r3.valid;
    }

    // Propagate ready backward to determine whether a new product can enter.
    bool in_ready(bool out_ready) const {
        bool s3_ready = out_ready || !r3.valid;
        bool s2_ready = s3_ready  || !r2.valid;
        bool s1_ready = s2_ready  || !r1.valid;
        return s1_ready;
    }

    // Packed product from r3. Caller should check out_valid() first.
    const uint32_t& out_data() const {
        return r3.result;
    }

    // Metadata associated with the current product.
    const TensorCoreMeta& out_meta() const {
        return r3.meta;
    }

    // Advance one cycle. Registers update r3 -> r2 -> r1 to model a clock edge.
    void tick(bool out_ready, const Config& g_cfg, uint32_t& c_in) {

        // Compute per-stage ready signals.
        bool s3_ready = out_ready || !r3.valid;
        bool s2_ready = s3_ready  || !r2.valid;
        bool s1_ready = s2_ready  || !r1.valid;

        // Stage 3: normalize, round, and pack the product.
        if (s3_ready) {
            if (r2.valid) {
                r3.result = fmul_s3(r2.s2_result, 8, 14);
                r3.c_in = r2.c_in;
                r3.valid = true;
                r3.meta = r2.meta;
            } else {
                r3.valid = false;
                r3.meta = {};
            }
        }

        // Stage 2: multiply significands or insert a bubble.
        if (s2_ready) {
            if (r1.valid) {
                r2.s2_result = fmul_s2(r1.a_fp9, r1.b_fp9, 8, 14, r1.s1_result);
                r2.c_in = r1.c_in;
                r2.valid = true;
                r2.meta = r1.meta;
            } else {
                r2.valid = false;
                r2.meta = {};
            }
        }

        // Stage 1: expand operands, compute exponent state, or insert a bubble.
        if (s1_ready) {
            if (mul_input.input_valid) {
                r1.a_fp9 = static_cast<uint16_t>(mul_input.a_fp9);
                r1.b_fp9 = static_cast<uint16_t>(mul_input.b_fp9);
                r1.c_in = c_in;
                r1.s1_result = fmul_s1(r1.a_fp9, r1.b_fp9, 8, 14, g_cfg.rm);
                r1.valid = true;
                r1.meta = mul_input.meta;
            } else {
                r1.valid = false;
                r1.meta = {};
            }
        }
    }
};
