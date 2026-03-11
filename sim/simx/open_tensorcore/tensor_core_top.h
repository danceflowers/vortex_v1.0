#pragma once
#include <cstdint>
#include "config_register.h"
#include "tc_mul_add.h"
#include "fp_types.h"
struct TensorCoreTop {
    static constexpr int M = 8, K = 8, N = 8;

    // Input data (all K elements arrive simultaneously)
    uint16_t a_in[M][K];  // A matrix in FP9
    uint16_t b_in[K][N];  // B matrix in FP9 (transposed for column access)
    uint32_t c_in[M][N]; // C matrix in FP22
    int jobs_completed = 0;
    int set_jobs = 0;
    bool input_loaded = false;
    int cycle_count = 0;
    // Output
    //uint32_t d_fp22[M][N]; // Raw FP22 results
    uint16_t d_out[M][N];  // Final output bits (FP8/FP16/FP32)
    bool     d_valid[M][N];

    
    tc_mul_add tc_dot_product[M][N];

    void reset(){
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                tc_dot_product[i][j].reset();
                d_valid[i][j] = false;
            }
 
        input_loaded = false;
        cycle_count = 0;
        set_jobs = 1;
        jobs_completed = 0;
    }

void load_inputs(const uint16_t a[M][K], const uint16_t b[K][N],
                     const uint32_t c[M][N] ) {

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < K; k++) {
                tc_dot_product[i][j].mul_add_input.a_in[k] = a[i][k];
                tc_dot_product[i][j].mul_add_input.b_in[k] = b[k][j];
                }    
                tc_dot_product[i][j].mul_add_input.prec = g_cfg.precisions.at(0);
                tc_dot_product[i][j].mul_add_input.c_in = c[i][j];
                tc_dot_product[i][j].mul_add_input.input_valid = true;
            }
        }
    }

    void load_invalid() {

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                tc_dot_product[i][j].mul_add_input.input_valid = false;
            }
        }
    }


        void tick(bool out_ready) {
        cycle_count++;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                tc_dot_product[i][j].tick(out_ready ,g_cfg);
            }
        }
    }

        bool run() {

        bool all_done = false;
            tick(true);
            all_done = true;
            for (int i = 0; i < M && all_done; i++)
                for (int j = 0; j < N && all_done; j++)
                    if (!tc_dot_product[i][j].out_valid()) all_done = false;
            if (all_done) {
                jobs_completed++;
                for (int i = 0; i < M; i++)
                    for (int j = 0; j < N; j++) {
                        d_out[i][j] = tc_dot_product[i][j].out_data();
                        d_valid[i][j] = true;
                    }
        }
        

        return all_done;
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