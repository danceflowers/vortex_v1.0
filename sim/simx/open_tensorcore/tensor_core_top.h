#pragma once

#include <cstdint>
#include "config_register.h"
#include "tc_mul_add.h"
#include "fp_types.h"

struct TensorCoreRetire {
    static constexpr int M = 8;
    static constexpr int N = 8;

    TensorCoreMeta meta;
    uint32_t fp22_out[M][N];
    uint32_t d_out[M][N];
    bool valid = false;

    void reset() {
        meta = {};
        valid = false;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                fp22_out[i][j] = 0;
                d_out[i][j] = 0;
            }
        }
    }
};

struct TensorCoreTop {
    static constexpr int M = 8, K = 8, N = 8;

    uint16_t a_in[M][K];
    uint16_t b_in[K][N];
    uint32_t c_in[M][N];
    uint32_t fp22_out[M][N];
    uint32_t d_out[M][N];
    bool d_valid[M][N];

    int jobs_completed = 0;
    int set_jobs = 0;
    bool input_loaded = false;
    int cycle_count = 0;

    tc_mul_add tc_dot_product[M][N];

    TensorCoreMeta staged_meta;
    bool staged_valid = false;
    TensorCoreRetire retired;

    void reset(){
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                tc_dot_product[i][j].reset();
                fp22_out[i][j] = 0;
                d_out[i][j] = 0;
                d_valid[i][j] = false;
                c_in[i][j] = 0;
            }
        }
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                a_in[i][k] = 0;
            }
        }
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < N; ++j) {
                b_in[k][j] = 0;
            }
        }
        staged_meta = {};
        staged_valid = false;
        retired.reset();
        input_loaded = false;
        cycle_count = 0;
        set_jobs = 1;
        jobs_completed = 0;
    }

    bool ready(bool out_ready = true) const {
        if (staged_valid) {
            return false;
        }
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                if (!tc_dot_product[i][j].in_ready(out_ready)) {
                    return false;
                }
            }
        }
        return true;
    }

    void push_uop(const uint16_t a[M][K],
                  const uint16_t b[K][N],
                  const uint32_t c[M][N],
                  const TensorCoreMeta& meta) {
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                a_in[i][k] = a[i][k];
            }
        }
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < N; ++j) {
                b_in[k][j] = b[k][j];
            }
        }
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                c_in[i][j] = c[i][j];
            }
        }
        staged_meta = meta;
        staged_meta.valid = true;
        staged_valid = true;
        input_loaded = true;
    }

    void load_inputs(const uint16_t a[M][K],
                     const uint16_t b[K][N],
                     const uint32_t c[M][N]) {
        push_uop(a, b, c, TensorCoreMeta{});
    }

    void load_invalid() {
        staged_valid = false;
        input_loaded = false;
        staged_meta = {};
    }

    bool pop_retired(TensorCoreRetire* out) {
        if (!retired.valid) {
            return false;
        }
        if (out != nullptr) {
            *out = retired;
        }
        retired.valid = false;
        return true;
    }

    void tick(bool out_ready) {
        cycle_count++;
        retired.valid = false;

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                for (int k = 0; k < K; ++k) {
                    tc_dot_product[i][j].mul_add_input.a_in[k] = a_in[i][k];
                    tc_dot_product[i][j].mul_add_input.b_in[k] = b_in[k][j];
                }
                tc_dot_product[i][j].mul_add_input.prec = g_cfg.precisions.at(0);
                tc_dot_product[i][j].mul_add_input.c_in = c_in[i][j];
                tc_dot_product[i][j].mul_add_input.input_valid = staged_valid;
                tc_dot_product[i][j].mul_add_input.meta = staged_valid ? staged_meta : TensorCoreMeta{};
            }
        }

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                tc_dot_product[i][j].tick(out_ready ,g_cfg);
            }
        }

        staged_valid = false;
        input_loaded = false;
        staged_meta = {};

        bool all_done = true;
        for (int i = 0; i < M && all_done; i++) {
            for (int j = 0; j < N && all_done; j++) {
                if (!tc_dot_product[i][j].out_valid()) {
                    all_done = false;
                }
            }
        }

        if (all_done) {
            jobs_completed++;
            retired.valid = true;
            retired.meta = tc_dot_product[0][0].out_meta();
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    fp22_out[i][j] = tc_dot_product[i][j].out_fp22();
                    d_out[i][j] = tc_dot_product[i][j].out_data();
                    d_valid[i][j] = true;
                    retired.fp22_out[i][j] = fp22_out[i][j];
                    retired.d_out[i][j] = d_out[i][j];
                }
            }
        } else {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    d_valid[i][j] = false;
                }
            }
        }
    }

    bool run() {
        tick(true);
        return retired.valid;
    }
};

// =============================================================================
// Functional (non-pipelined) reference: compute D = A*B + C using same arithmetic
// =============================================================================
inline void reference_matmul(const uint16_t a_fp9[8][8], const uint16_t b_fp9[8][8],
                              const uint32_t c_fp22[8][8], double d_fp[8][8],
                              RoundingMode rm = RNE)
{
    (void)rm;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            double acc = 0.0;
            for (int k = 0; k < 8; k++) {
                double a = fp9_to_double(a_fp9[i][k]);
                double b = fp9_to_double(b_fp9[k][j]);
                acc += a * b;
            }
            d_fp[i][j] = acc + fp22_to_double(c_fp22[i][j]);
        }
    }
}
