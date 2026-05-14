#pragma once
#include <array>
#include <cstdlib>
#include <cstdint>
#include "fp_types.h"
#include "tc_mul_pipe.h"
#include "tc_add_pipe.h"
#include "tc_add4_pipe_fp22.h"

// ============================================================================
// tc_mul_add -- one scalar dot-product lane in the 8x8 TensorCore array.
// ============================================================================
//
// This lane computes one output element:
//   D[i][j] = sum_k(A[i][k] * B[k][j]) + C[i][j]
//
// Pipeline:
//   - 8 parallel mul_pipe instances produce FP22 products.
//   - two add4_pipe_fp22 instances reduce four products each.
//   - merge_add combines the partial sums.
//   - final_add adds either the passthrough C value or the resident FIFO value.
//
// In non-resident mode, c_in is carried through the pipeline as passthrough
// data. In output-resident mode, accum_fifo supplies C to final_add and accepts
// the feedback result, allowing repeated accumulation inside the lane.
// ============================================================================

struct tc_mul_add {

    // 2-entry elastic accumulator FIFO used by output-resident mode.
    struct accum_fifo_t {
        struct entry_t {
            uint32_t data = 0;
            bool valid = false;
        };

        entry_t curr;   // FIFO head, consumed by final_add.
        entry_t next;   // FIFO tail, used for prefetch/feedback buffering.

        void reset() {
            curr = {};
            next = {};
        }

        bool can_pop() const { return curr.valid; }

        bool can_push(bool consuming) const {
            return !next.valid || (consuming && next.valid);
        }

        uint32_t peek() const { return curr.data; }

        // Advance pop/push in one tick while preserving FIFO order.
        void advance(bool consumed, bool push_valid, uint32_t push_data) {
            auto old_curr = curr;
            auto old_next = next;

            if (consumed) {
                if (old_next.valid) {
                    curr = old_next;
                    next = push_valid ? entry_t{push_data, true} : entry_t{};
                } else {
                    curr = {};
                    next = push_valid ? entry_t{push_data, true} : entry_t{};
                }
                return;
            }

            if (push_valid) {
                if (!old_curr.valid) {
                    curr = {push_data, true};
                } else if (!old_next.valid) {
                    next = {push_data, true};
                }
            }
        }

        bool empty() const { return !curr.valid && !next.valid; }
    } accum_fifo;

    // Arithmetic sub-pipelines.

    std::array<mul_pipe, 8> mul_array;
    std::array<add4_pipe_fp22, 2> add4_level;
    add_pipe_fp22 merge_add;
    add_pipe_fp22 final_add;

    // Input latch written by TensorCoreTop before tick().
    struct {
        uint32_t a_in[8];       // A[i][k], k = 0..7, in FP9 format.
        uint32_t b_in[8];       // B[k][j], k = 0..7, in FP9 format.
        uint32_t c_in;          // Passthrough C value for non-resident mode.
        PrecisionType prec;
        bool input_valid;
        TensorCoreMeta meta;
    } mul_add_input;

    // Final output register.
    struct {
        uint32_t fp22_result;
        bool valid;
        TensorCoreMeta meta;
    } r2;

    // Pulses when resident mode actually consumes one FIFO value.
    bool resident_turnover_pulse = false;

    // Reset all arithmetic state and resident FIFO contents.
    void reset() {
        for (int i = 0; i < 8; i++) mul_array[i].reset();
        for (int i = 0; i < 2; i++) add4_level[i].reset();
        merge_add.reset();
        final_add.reset();
        accum_fifo.reset();
        r2.fp22_result = 0;
        r2.valid = false;
        r2.meta = {};
        mul_add_input.meta = {};
        mul_add_input.input_valid = false;
        resident_turnover_pulse = false;
    }

    // Public status/query interface.

    bool out_valid() const { return r2.valid; }

    bool active() const {
        if (mul_add_input.input_valid || r2.valid) return true;
        for (const auto& mul : mul_array) { if (mul.active()) return true; }
        for (const auto& add : add4_level) { if (add.active()) return true; }
        if (merge_add.active() || final_add.active()) return true;
        if (!accum_fifo.empty()) return true;
        return false;
    }

    // True while arithmetic pipeline stages hold work. Resident FIFO contents
    // waiting for drain are not counted as active arithmetic.
    bool pipeline_active() const {
        if (mul_add_input.input_valid || r2.valid) return true;
        for (const auto& mul : mul_array) { if (mul.active()) return true; }
        for (const auto& add : add4_level) { if (add.active()) return true; }
        if (merge_add.active() || final_add.active()) return true;
        return false;
    }

    bool in_ready(bool out_ready) const {
        bool s8_ready = out_ready || !r2.valid;
        bool s7_ready = final_add.in_ready(s8_ready);
        bool s6_ready = merge_add.in_ready(s7_ready);
        bool s5_ready = true;
        for (int i = 0; i < 2; ++i) s5_ready &= add4_level[i].in_ready(s6_ready);
        bool s4_ready = true;
        for (int i = 0; i < 8; ++i) s4_ready &= mul_array[i].in_ready(s5_ready);
        return s4_ready;
    }

    const uint32_t& out_fp22() const { return r2.fp22_result; }
    const TensorCoreMeta& out_meta() const { return r2.meta; }

    // Load prefetched C values into the resident accumulator FIFO.
    bool load_fifo(uint32_t fp22_value) {
        if (!accum_fifo.curr.valid) {
            accum_fifo.curr = {fp22_value, true};
            return true;
        }
        if (!accum_fifo.next.valid) {
            accum_fifo.next = {fp22_value, true};
            return true;
        }
        return false;
    }

    // Advance the full scalar lane by one cycle.
    // circulating selects resident FIFO accumulation instead of passthrough C.
    void tick(bool out_ready,
              const Config& g_cfg,
              bool circulating = false,
              bool resident_tail_to_dmem_valid = false,
              uint32_t resident_tail_to_dmem_async_id = 0) {
        resident_turnover_pulse = false;

        // Ready chain from output register back to multiplier inputs.
        bool s8_ready = out_ready || !r2.valid;
        bool s7_ready = final_add.in_ready(s8_ready);
        bool s6_ready = merge_add.in_ready(s7_ready);
        bool s5_ready = true;
        for (int i = 0; i < 2; ++i) s5_ready &= add4_level[i].in_ready(s6_ready);
        bool s4_ready = true;
        for (int i = 0; i < 8; ++i) s4_ready &= mul_array[i].in_ready(s5_ready);

        // ======== s8: final_add → r2 ========
        bool r2_was_valid = false;
        uint32_t r2_feedback_data = 0;

        if (s8_ready) {
            if (final_add.out_valid()) {
                r2.fp22_result = final_add.out_data();
                r2.valid = true;
                r2.meta = final_add.out_meta();
            } else {
                r2.valid = false;
                r2.meta = {};
            }
        }
        r2_was_valid = r2.valid;
        r2_feedback_data = r2.fp22_result;

        // s7: feed merge_add and the selected C operand into final_add.
        //
        // circulating=true  selects accum_fifo.peek().
        // circulating=false selects merge_add.out_passthrough().
        bool merge_valid = merge_add.out_valid();
        bool fifo_consumed = false;

        if (s7_ready) {
            if (circulating) {
                // Resident mode: source C from the accumulator FIFO.
                bool fifo_ready = accum_fifo.can_pop();
                if (merge_valid && fifo_ready) {
                    final_add.add_input.a_in = merge_add.out_data();
                    final_add.add_input.b_in = accum_fifo.peek();
                    final_add.add_input.input_valid = true;
                    final_add.add_input.meta = merge_add.out_meta();
                    fifo_consumed = true;
                } else {
                    final_add.add_input.input_valid = false;
                    final_add.add_input.meta = {};
                }
            } else {
                // Non-resident mode: source C from the passthrough chain.
                if (merge_valid) {
                    final_add.add_input.a_in = merge_add.out_data();
                    final_add.add_input.b_in = merge_add.out_passthrough();
                    final_add.add_input.input_valid = true;
                    final_add.add_input.meta = merge_add.out_meta();
                } else {
                    final_add.add_input.input_valid = false;
                    final_add.add_input.meta = {};
                }
            }
            final_add.tick(s8_ready, g_cfg);
        }

        // Advance the resident FIFO only when resident mode is active.
        if (circulating) {
            bool tail_to_dmem = resident_tail_to_dmem_valid
                             && r2_was_valid
                             && (r2.meta.async_id == resident_tail_to_dmem_async_id);
            bool push_valid = r2_was_valid && !tail_to_dmem;
            accum_fifo.advance(fifo_consumed, push_valid, r2_feedback_data);
            resident_turnover_pulse = fifo_consumed;
        }

        // ======== s6: add4 → merge_add ========
        bool r5_valid = true;
        for (int i = 0; i < 2; ++i) r5_valid &= add4_level[i].out_valid();

        if (s6_ready) {
            if (r5_valid) {
                merge_add.add_input.a_in = add4_level[0].out_data();
                merge_add.add_input.b_in = add4_level[1].out_data();
                merge_add.add_input.input_valid = true;
                merge_add.add_input.meta = add4_level[0].out_meta();
            } else {
                merge_add.add_input.input_valid = false;
                merge_add.add_input.meta = {};
            }
            merge_add.tick(s7_ready, g_cfg, add4_level[0].out_passthrough());
        }

        // ======== s5: mul → add4 ========
        bool r4_valid = true;
        for (int i = 0; i < 8; ++i) r4_valid &= mul_array[i].out_valid();

        if (s5_ready) {
            if (r4_valid) {
                for (int i = 0; i < 2; ++i) {
                    const int base = i * 4;
                    add4_level[i].add_input.in[0] = mul_array[base + 0].out_data();
                    add4_level[i].add_input.in[1] = mul_array[base + 1].out_data();
                    add4_level[i].add_input.in[2] = mul_array[base + 2].out_data();
                    add4_level[i].add_input.in[3] = mul_array[base + 3].out_data();
                    add4_level[i].add_input.input_valid = true;
                    add4_level[i].add_input.meta = mul_array[base].out_meta();
                    add4_level[i].tick(s6_ready, g_cfg, mul_array[base].r3.c_in);
                }
            } else {
                for (int i = 0; i < 2; ++i) {
                    const int base = i * 4;
                    add4_level[i].add_input.input_valid = false;
                    add4_level[i].add_input.meta = {};
                    add4_level[i].tick(s6_ready, g_cfg, mul_array[base].r3.c_in);
                }
            }
        }

        // s4: drive external input into the multiplier array.
        if (s4_ready) {
            if (mul_add_input.input_valid) {
                for (int i = 0; i < 8; ++i) {
                    mul_array[i].mul_input.a_fp9 = mul_add_input.a_in[i];
                    mul_array[i].mul_input.b_fp9 = mul_add_input.b_in[i];
                    mul_array[i].mul_input.input_valid = true;
                    mul_array[i].mul_input.meta = mul_add_input.meta;
                    mul_array[i].tick(s5_ready, g_cfg, mul_add_input.c_in);
                }
            } else {
                for (int i = 0; i < 8; ++i) {
                    mul_array[i].mul_input.input_valid = false;
                    mul_array[i].mul_input.meta = {};
                    mul_array[i].tick(s5_ready, g_cfg, mul_add_input.c_in);
                }
            }
        }
    }
};
