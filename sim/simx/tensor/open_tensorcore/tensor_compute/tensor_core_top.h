#pragma once

#include <cstdint>
#include "config_register.h"
#include "tc_mul_add.h"
#include "fp_types.h"
#include "tensor_cfg.h"

// Retired output for one 8x8 primitive.
struct TensorCoreRetire {
    static constexpr int M = 8;
    static constexpr int N = 8;

    TensorCoreMeta meta;
    uint32_t fp22_out[M][N];
    bool valid = false;

    void reset() {
        meta = {};
        valid = false;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                fp22_out[i][j] = 0;
    }
};

// Top-level 8x8 TensorCore primitive array.
//
// Accepts one 8x8 A/B/C primitive, broadcasts it to 64 scalar tc_mul_add lanes,
// and retires one 8x8 FP22 subtile result. C is carried through the pipeline as
// passthrough data alongside the dot-product reduction.
struct TensorCoreTop {
    static constexpr int M = 8, K = 8, N = 8;

    // Input staging buffers.
    uint16_t a_in[M][K];
    uint16_t b_in[K][N];
    uint32_t c_in[M][N];           // C bypass data for non-resident mode.
    uint32_t fp22_out[M][N];

    // Control and progress state.
    int jobs_completed = 0;
    int set_jobs = 0;
    bool input_loaded = false;
    int cycle_count = 0;

    // 8x8 compute array.
    tc_mul_add tc_dot_product[M][N];

    // One-cycle input metadata staging.
    TensorCoreMeta staged_meta;
    uint32_t staged_sparsity_kind = vortex::tensor::sparse_none;
    uint16_t staged_sparse_row_meta[M] = {};
    bool staged_valid = false;

    // Retire buffer.
    TensorCoreRetire retired;

    // =========================================================================
    void reset() {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                tc_dot_product[i][j].reset();
                fp22_out[i][j] = 0;
                c_in[i][j] = 0;
            }
        }
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k)
                a_in[i][k] = 0;
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
                b_in[k][j] = 0;
        staged_meta = {};
        staged_sparsity_kind = vortex::tensor::sparse_none;
        for (int i = 0; i < M; ++i)
            staged_sparse_row_meta[i] = 0;
        staged_valid = false;
        retired.reset();
        input_loaded = false;
        cycle_count = 0;
        set_jobs = 1;
        jobs_completed = 0;
    }

    bool ready(bool out_ready = true) const {
        if (staged_valid) return false;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                if (!tc_dot_product[i][j].in_ready(out_ready))
                    return false;
        return true;
    }

    bool active() const {
        if (staged_valid || retired.valid) return true;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                if (tc_dot_product[i][j].active())
                    return true;
        return false;
    }

    // Stage one primitive input.
    void push_uop(const uint16_t a[M][K],
                  const uint16_t b[K][N],
                  const uint32_t c[M][N],
                  const TensorCoreMeta& meta) {
        uint16_t sparse_row_meta[M] = {};
        push_uop(a, b, c, meta, vortex::tensor::sparse_none, sparse_row_meta);
    }

    void push_uop(const uint16_t a[M][K],
                  const uint16_t b[K][N],
                  const uint32_t c[M][N],
                  const TensorCoreMeta& meta,
                  uint32_t sparsity_kind,
                  const uint16_t sparse_row_meta[M]) {
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k)
                a_in[i][k] = a[i][k];
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
                b_in[k][j] = b[k][j];
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                c_in[i][j] = c[i][j];
        staged_meta = meta;
        staged_meta.valid = true;
        staged_sparsity_kind = sparsity_kind;
        for (int i = 0; i < M; ++i)
            staged_sparse_row_meta[i] = sparse_row_meta[i];
        staged_valid = true;
        input_loaded = true;
    }

    bool pop_retired(TensorCoreRetire* out) {
        if (!retired.valid) return false;
        if (out != nullptr) *out = retired;
        retired.valid = false;
        return true;
    }

    // Advance all 64 lanes by one cycle and collect a retired primitive.
    void tick(bool out_ready) {
        cycle_count++;
        retired.valid = false;

        // Stage 1: broadcast the staged primitive to all 64 lanes.
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < K; ++k) {
                    tc_dot_product[i][j].mul_add_input.a_in[k] = a_in[i][k];
                    tc_dot_product[i][j].mul_add_input.b_in[k] = b_in[k][j];
                }
                tc_dot_product[i][j].mul_add_input.c_in = c_in[i][j];
                tc_dot_product[i][j].mul_add_input.sparsity_kind =
                    staged_valid ? staged_sparsity_kind : vortex::tensor::sparse_none;
                tc_dot_product[i][j].mul_add_input.sparse_row_meta =
                    staged_valid ? staged_sparse_row_meta[i] : 0;
                tc_dot_product[i][j].mul_add_input.input_valid = staged_valid;
                tc_dot_product[i][j].mul_add_input.meta =
                    staged_valid ? staged_meta : TensorCoreMeta{};
            }
        }

        // Stage 2: advance all 64 lanes by one cycle.
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                tc_dot_product[i][j].tick(out_ready, g_cfg);

        // Stage 3: clear the one-cycle staging registers.
        staged_valid = false;
        input_loaded = false;
        staged_meta = {};
        staged_sparsity_kind = vortex::tensor::sparse_none;
        for (int i = 0; i < M; ++i)
            staged_sparse_row_meta[i] = 0;

        // Stage 4: retire once every lane produced a valid result.
        bool all_done = true;
        for (int i = 0; i < M && all_done; i++)
            for (int j = 0; j < N && all_done; j++)
                if (!tc_dot_product[i][j].out_valid())
                    all_done = false;

        if (all_done) {
            jobs_completed++;
            retired.valid = true;
            retired.meta = tc_dot_product[0][0].out_meta();
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++) {
                    fp22_out[i][j] = tc_dot_product[i][j].out_fp22();
                    retired.fp22_out[i][j] = fp22_out[i][j];
                }
        }
    }
};
