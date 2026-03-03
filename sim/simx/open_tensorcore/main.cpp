
#include "tensor_core_top.h"
#include "config_register.h"
#include "fp_types.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
static uint32_t rng_state = 42;
inline uint32_t xorshift32() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

inline double rand_double(double lo, double hi) {
    return lo + (hi - lo) * (double)(xorshift32() & 0xFFFF) / 65535.0;
}

struct MatrixSet {
    uint16_t a_fp9[8][8];
    uint16_t b_fp9[8][8];
    uint32_t c_fp22[8][8];
    uint16_t a_raw[8][8];
    uint16_t b_raw[8][8];
    uint32_t c_raw[8][8];
};

double raw_to_double(uint32_t bits, PrecisionType prec) {
    switch (prec) {
        case PREC_FP4_E2M1: return fp4_to_double((uint8_t)(bits & 0xF));
        case PREC_FP8_E4M3: return fp8_e4m3_to_double((uint8_t)(bits & 0xFF));
        case PREC_FP8_E5M2: return fp8_e5m2_to_double((uint8_t)(bits & 0xFF));
        case PREC_FP16:     return fp16_to_double((uint16_t)(bits & 0xFFFF));
        case PREC_FP32: {
            float f = 0.0f;
            uint32_t raw = bits;
            std::memcpy(&f, &raw, sizeof(float));
            return (double)f;
        }
        default: return 0.0;
    }
}

MatrixSet generate_random_matrices(PrecisionType prec) {
    MatrixSet ms = {};
    double range_lo, range_hi;

    switch (prec) {
        case PREC_FP4_E2M1: range_lo = -3.0/10; range_hi = 3.0/10; break;
        case PREC_FP8_E4M3: range_lo = -8.0/10; range_hi = 8.0/10; break;
        case PREC_FP8_E5M2: range_lo = -4.0/10; range_hi = 4.0/10; break;
        case PREC_FP16:     range_lo = -10.0/10; range_hi = 10.0/10; break;
        default:            range_lo = -1.0/10; range_hi = 1.0/10; break;
    }

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
 
            // double va = rand_double(range_lo, range_hi);
            // double vb = rand_double(range_lo, range_hi);
            // double vc = rand_double(range_lo * 0.5, range_hi * 0.5);
            double va = (rand() % 200 - 100) / 100.0;
            double vb = (rand() % 200 - 100) / 100.0;
            double vc = (rand() % 200 - 100) / 100.0;
            switch (prec) {
                case PREC_FP4_E2M1:
                    ms.a_raw[i][j] = double_to_fp4(va);
                    ms.b_raw[i][j] = double_to_fp4(vb);
                    ms.c_raw[i][j] = double_to_fp16(vc);
                    ms.a_fp9[i][j] = fp4_to_fp9(ms.a_raw[i][j]);
                    ms.b_fp9[i][j] = fp4_to_fp9(ms.b_raw[i][j]);
                    ms.c_fp22[i][j] = fp16_to_fp22(ms.c_raw[i][j]);
                    break;
                case PREC_FP8_E4M3:
                    ms.a_raw[i][j] = double_to_fp8_e4m3(va);
                    ms.b_raw[i][j] = double_to_fp8_e4m3(vb);
                    ms.c_raw[i][j] = double_to_fp16(vc);
                    ms.a_fp9[i][j] = fp8_e4m3_to_fp9(ms.a_raw[i][j]);
                    ms.b_fp9[i][j] = fp8_e4m3_to_fp9(ms.b_raw[i][j]);
                    ms.c_fp22[i][j] = fp16_to_fp22(ms.c_raw[i][j]);
                    break;
                case PREC_FP8_E5M2:
                    ms.a_raw[i][j] = double_to_fp8_e5m2(va);
                    ms.b_raw[i][j] = double_to_fp8_e5m2(vb);
                    ms.c_raw[i][j] = double_to_fp16(vc);
                    ms.a_fp9[i][j] = fp8_e5m2_to_fp9(ms.a_raw[i][j]);
                    ms.b_fp9[i][j] = fp8_e5m2_to_fp9(ms.b_raw[i][j]);
                    ms.c_fp22[i][j] = fp16_to_fp22(ms.c_raw[i][j]);
                    break;
                case PREC_FP16:
                    ms.a_raw[i][j] = double_to_fp16(va);
                    ms.b_raw[i][j] = double_to_fp16(vb);
                    ms.c_raw[i][j] = double_to_fp16(vc);
                    ms.a_fp9[i][j] = fp16_to_fp9(ms.a_raw[i][j]);
                    ms.b_fp9[i][j] = fp16_to_fp9(ms.b_raw[i][j]);
                    ms.c_fp22[i][j] = fp16_to_fp22(ms.c_raw[i][j]);
                    break;
                default: break;
            }
        }
    }
    return ms;
}

// =============================================================================
// Precision name string
// =============================================================================
const char* prec_name(PrecisionType p) {
    switch (p) {
        case PREC_FP4_E2M1: return "FP4_E2M1";
        case PREC_FP8_E4M3: return "FP8_E4M3";
        case PREC_FP8_E5M2: return "FP8_E5M2";
        case PREC_FP16:     return "FP16";
        case PREC_FP32:     return "FP32";
        default:            return "UNKNOWN";
    }
}



double output_bits_to_double(uint32_t bits, PrecisionType output_prec) {
    switch (output_prec) {
        case PREC_FP8_E4M3: return fp8_e4m3_to_double((uint8_t)(bits & 0xFF));
        case PREC_FP8_E5M2: return fp8_e5m2_to_double((uint8_t)(bits & 0xFF));
        case PREC_FP16:     return fp16_to_double((uint16_t)(bits & 0xFFFF));
        case PREC_FP32: {
            float f = 0.0f;
            uint32_t raw = bits;
            std::memcpy(&f, &raw, sizeof(float));
            return (double)f;
        }
        case PREC_FP4_E2M1:
        default:            return 0.0;
    }
}

void print_matrix_double(const char* title, const double m[8][8]) {
    printf("    %s\n", title);
    for (int i = 0; i < 8; i++) {
        printf("      ");
        for (int j = 0; j < 8; j++) {
            printf("%9.4f ", m[i][j]);
        }
        printf("\n");
    }
}


void print_matrix_output(const char* title, const uint32_t m[8][8], PrecisionType out_prec) {
    printf("    %s\n", title);
    for (int i = 0; i < 8; i++) {
        printf("      ");
        for (int j = 0; j < 8; j++) {
            printf("%9.4f ", output_bits_to_double(m[i][j], out_prec));
        }
        printf("\n");
    }
}

void golden_fp32_matmul(const MatrixSet& ms, PrecisionType in_prec, double out[8][8]) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            float acc = 0.0f;
            for (int k = 0; k < 8; k++) {
                float a = (float)raw_to_double(ms.a_raw[i][k], in_prec);
                float b = (float)raw_to_double(ms.b_raw[k][j], in_prec);
                acc += a * b;
            }
            float c = (float)raw_to_double(ms.c_raw[i][j], in_prec);
            out[i][j] = (double)(acc + c);
        }
    }
}


const char* rm_name(RoundingMode rm) {
    switch (rm) {
        case RNE: return "RNE (Round to Nearest, ties to Even)";
        case RTZ: return "RTZ (Round toward Zero)";
        case RDN: return "RDN (Round Down / toward -Inf)";
        case RUP: return "RUP (Round Up / toward +Inf)";
        case RMM: return "RMM (Round to Nearest, ties to Max Magnitude)";
        default:  return "UNKNOWN";
    }
}
// =============================================================================
// Test 2: Back-to-back pipelined matmuls
// =============================================================================
void test_pipelined_throughput() {

    TensorCoreTop sim;
    sim.reset();

    struct JobResult {
        uint32_t d_out[8][8];
        double ref[8][8];
        int cycles;
    };
    
     int num_jobs = 8;
    std::vector<JobResult> results;
    int total_cycles = 0;
    sim.reset();
     for (int job = 0; job < num_jobs; job++) {
        MatrixSet ms = generate_random_matrices(g_cfg.precisions[0]);
        JobResult jr;
        reference_matmul(ms.a_fp9, ms.b_fp9, ms.c_fp22, jr.ref, g_cfg.rm);
        results.push_back(jr);

        sim.load_inputs(ms.a_raw, ms.b_raw, ms.c_raw);
        if (sim.run()) {
            auto& done_jr = results[sim.jobs_completed - 1];
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 8; j++)
                    done_jr.d_out[i][j] = sim.d_out[i][j];
        }
     }
     sim.load_invalid();
    while (sim.jobs_completed < num_jobs)
     {
        if (sim.run()){
            auto& jr = results[sim.jobs_completed - 1];
             for (int i = 0; i < 8; i++)
                 for (int j = 0; j < 8; j++)
                     jr.d_out[i][j] = sim.d_out[i][j];
        }
     }

    for (int job = 0; job < num_jobs; job++) {
        auto& jr = results[job];
        print_matrix_double("REF", jr.ref);
        print_matrix_output("OUTPUT", jr.d_out, PREC_FP8_E5M2);
    }
        

    printf("\n  Total: %d cycles for %d jobs | %.1f cycles/matmul ",
           sim.cycle_count, num_jobs, sim.cycle_count / (double)num_jobs);
}
// =============================================================================
// Argument parsing
// =============================================================================
PrecisionType parse_precision(const char* s) {
    if (strcmp(s, "FP4_E2M1") == 0 || strcmp(s, "FP4") == 0)     return PREC_FP4_E2M1;
    if (strcmp(s, "FP8_E4M3") == 0 || strcmp(s, "E4M3") == 0)    return PREC_FP8_E4M3;
    if (strcmp(s, "FP8_E5M2") == 0 || strcmp(s, "E5M2") == 0)    return PREC_FP8_E5M2;
    if (strcmp(s, "FP16") == 0)                                    return PREC_FP16;
    if (strcmp(s, "FP32") == 0)                                    return PREC_FP32;
    fprintf(stderr, "  Error: Unknown precision '%s'\n", s);
    fprintf(stderr, "  Valid: FP4_E2M1 | FP8_E4M3 | FP8_E5M2 | FP16 | FP32\n\n");
    exit(1);
}

RoundingMode parse_rounding(const char* s) {
    if (strcmp(s, "RNE") == 0) return RNE;
    if (strcmp(s, "RTZ") == 0) return RTZ;
    if (strcmp(s, "RDN") == 0) return RDN;
    if (strcmp(s, "RUP") == 0) return RUP;
    if (strcmp(s, "RMM") == 0) return RMM;
    fprintf(stderr, "  Error: Unknown rounding mode '%s'\n", s);
    fprintf(stderr, "  Valid: RNE | RTZ | RDN | RUP | RMM\n\n");
    exit(1);
}

bool parse_args(int argc, char* argv[]) {
    g_cfg.precisions.clear();
    g_cfg.out_precisions.clear();
    g_cfg.test_id  = 0;
    g_cfg.rm       = RNE;
    g_cfg.seed     = 0;
    g_cfg.show_help = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            g_cfg.show_help = true;
            return true;
        } else if (strcmp(argv[i], "--prec") == 0 && i + 1 < argc) {
            g_cfg.precisions.push_back(parse_precision(argv[++i]));
        } else if (strcmp(argv[i], "--test") == 0 && i + 1 < argc) {
            g_cfg.test_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--rm") == 0 && i + 1 < argc) {
            g_cfg.rm = parse_rounding(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            g_cfg.seed = (uint32_t)atol(argv[++i]);
        } else {
            fprintf(stderr, "  Error: Unknown argument '%s'\n\n", argv[i]);
            return false;
        }
    }

    if (g_cfg.precisions.empty()) {
        g_cfg.precisions = {PREC_FP4_E2M1, PREC_FP8_E4M3, PREC_FP8_E5M2, PREC_FP16};
    }
    if (g_cfg.out_precisions.empty()) {
        g_cfg.out_precisions = {PREC_FP8_E4M3, PREC_FP8_E5M2, PREC_FP16, PREC_FP32};
    }

    return true;
}


static void print_summary() {}

// =============================================================================
// Main
// =============================================================================
int main(int argc, char* argv[]) {
    if (!parse_args(argc, argv)) {
        //print_usage(argv[0]);
        return 1;
    }

    rng_state = g_cfg.seed ? g_cfg.seed : (uint32_t)time(nullptr);


    //print_config();

    bool run_all = (g_cfg.test_id == 0);

    if (run_all) print_summary();

    if (run_all || g_cfg.test_id == 2) test_pipelined_throughput();

    return 0;
}
