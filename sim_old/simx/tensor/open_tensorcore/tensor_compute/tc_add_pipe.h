#pragma once
#include <cstdint>
#include "config_register.h"
#include "fadd_s1.h"
#include "fadd_s2.h"

// Combinational FP22 add helper. This folds fadd_s1/fadd_s2 into one call for
// storage helpers that need arithmetic behavior without pipeline timing.
inline uint32_t fp22_add_bits(uint32_t a_bits, uint32_t b_bits, RoundingMode rm) {
    return fadd_s2(fadd_s1(a_bits, b_bits, 8, 14, 14, rm), 8, 14);
}

// Two-stage floating-point adder used in the reduction tree. Stage 1 classifies
// and aligns operands, while stage 2 selects the near/far result and rounds it.
// c_in is a passthrough payload that keeps the C bypass value aligned with the
// arithmetic result under valid/ready backpressure.
struct add_pipe {

    // Input latch written by the owner before tick().
    struct {
        uint32_t a_in;           // Floating-point operand A.
        uint32_t b_in;           // Floating-point operand B.
        bool input_valid;        // Input valid bit.
        TensorCoreMeta meta;     // Metadata carried with the data.
    } add_input;

    // Pipeline registers.
    // Stage 1 register: fadd_s1 output plus passthrough metadata.
    struct {
        fadd_s1_out s1_result;   // Near/far path intermediate result.
        uint32_t c_in;           // Passthrough C bypass data.
        bool valid;              // Stage valid bit.
        TensorCoreMeta meta;     // Metadata carried with the data.
    } r1;

    // Stage 2 register: rounded add result plus passthrough metadata.
    struct {
        uint32_t result;         // Packed add result.
        uint32_t c_in;           // Passthrough C bypass data.
        bool valid;              // Stage valid bit.
        TensorCoreMeta meta;     // Metadata carried with the data.
    } r2;

    // Clear all valid bits and input metadata.
    void reset() {
        r1.valid = false;
        r2.valid = false;
        add_input.input_valid = false;
        add_input.meta = {};
    }

    // External query interface.
    // True when stage 2 holds a valid result.
    bool out_valid() const {
        return r2.valid;
    }

    // True when any input or pipeline stage holds work.
    bool active() const {
        return add_input.input_valid || r1.valid || r2.valid;
    }

    // Propagate ready backward from the output to decide if a new input fits.
    bool in_ready(bool out_ready) const {
        bool s2_ready = out_ready || !r2.valid;
        bool s1_ready = s2_ready || !r1.valid;
        return s1_ready;
    }

    // Packed add result from stage 2.
    const uint32_t& out_data() const {
        return r2.result;
    }

    // Metadata associated with the current output.
    const TensorCoreMeta& out_meta() const {
        return r2.meta;
    }

    // Advance one cycle. Stages update from output to input so newly captured
    // data cannot pass through multiple registers in the same tick.
    void tick(bool out_ready, const Config& g_cfg, uint32_t c_in) {
        bool s2_ready = out_ready || !r2.valid;
        bool s1_ready = s2_ready || !r1.valid;

        // Stage 2: round and pack the add result from r1.
        if (s2_ready) {
            if (r1.valid) {
                r2.result = fadd_s2(r1.s1_result, 5, 8);
                r2.c_in = r1.c_in;
                r2.valid = true;
                r2.meta = r1.meta;
            } else {
                r2.valid = false;
                r2.meta = {};
            }
        }

        // Stage 1: classify and align incoming operands.
        if (s1_ready) {
            if (add_input.input_valid) {
                r1.s1_result = fadd_s1(add_input.a_in, add_input.b_in, 5, 8, 8, g_cfg.rm);
                r1.c_in = c_in;
                r1.valid = true;
                r1.meta = add_input.meta;
            } else {
                r1.valid = false;
                r1.meta = {};
            }
        }
    }
};

// =====================================================================================
// add_pipe_fp22 -- two-stage FP22 adder used by merge/final accumulation.
// It has the same valid/ready timing as add_pipe, but runs fadd_s1/fadd_s2 with
// FP22 parameters. passthrough carries C or auxiliary data in lockstep with the
// arithmetic result.
// =====================================================================================
struct add_pipe_fp22 {

    // Input latch written by the owner before tick().
    struct {
        uint32_t a_in;           // FP22 operand A.
        uint32_t b_in;           // FP22 operand B.
        bool input_valid;        // Input valid bit.
        TensorCoreMeta meta;     // Metadata carried with the data.
    } add_input;

    // Pipeline registers.
    // ---------------------------------------------------------
    // Stage 1 register: FP22 fadd_s1 result plus passthrough metadata.
    struct {
        fadd_s1_out s1_result;   // Near/far path intermediate result.
        uint32_t passthrough;    // Payload aligned with the add result.
        bool valid;              // Stage valid bit.
        TensorCoreMeta meta;     // Metadata carried with the data.
    } r1;

    // Stage 2 register: rounded FP22 result plus passthrough metadata.
    struct {
        uint32_t result;         // Packed FP22 add result.
        uint32_t passthrough;    // Payload aligned with the add result.
        bool valid;              // Stage valid bit.
        TensorCoreMeta meta;     // Metadata carried with the data.
    } r2;

    // Clear all valid bits and passthrough payloads.
    void reset() {
        r1.valid = false;
        r1.passthrough = 0;
        r2.valid = false;
        r2.passthrough = 0;
        add_input.input_valid = false;
        add_input.meta = {};
    }

    // External query interface.
    // True when stage 2 holds a valid result.
    bool out_valid() const {
        return r2.valid;
    }

    // True when any input or pipeline stage holds work.
    bool active() const {
        return add_input.input_valid || r1.valid || r2.valid;
    }

    // Propagate ready backward from the output to the input.
    bool in_ready(bool out_ready) const {
        bool s2_ready = out_ready || !r2.valid;
        bool s1_ready = s2_ready || !r1.valid;
        return s1_ready;
    }

    // Packed FP22 add result from stage 2.
    const uint32_t& out_data() const {
        return r2.result;
    }

    // Metadata associated with the current output.
    const TensorCoreMeta& out_meta() const {
        return r2.meta;
    }

    // Passthrough payload aligned with the current output.
    uint32_t out_passthrough() const {
        return r2.passthrough;
    }

    // -------------------------------------------------------
    // Advance one cycle and keep passthrough data aligned with the FP22 result.
    void tick(bool out_ready, const Config& g_cfg, uint32_t passthrough = 0) {
        bool s2_ready = out_ready || !r2.valid;
        bool s1_ready = s2_ready || !r1.valid;

        // Stage 2: finish FP22 rounding from the stage-1 intermediate.
        if (s2_ready) {
            if (r1.valid) {
                r2.result = fadd_s2(r1.s1_result, 8, 14);
                r2.passthrough = r1.passthrough;
                r2.valid = true;
                r2.meta = r1.meta;
            } else {
                r2.passthrough = 0;
                r2.valid = false;
                r2.meta = {};
            }
        }

        // Stage 1: classify and align the incoming FP22 operands.
        if (s1_ready) {
            if (add_input.input_valid) {
                r1.s1_result = fadd_s1(add_input.a_in, add_input.b_in, 8, 14, 14, g_cfg.rm);
                r1.passthrough = passthrough;
                r1.valid = true;
                r1.meta = add_input.meta;
            } else {
                r1.passthrough = 0;
                r1.valid = false;
                r1.meta = {};
            }
        }
    }
};
