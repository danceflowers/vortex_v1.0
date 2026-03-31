
#include "tensor_core_top.h"
#include "config_register.h"
#include "fp_types.h"
#include "sparse_cmodel.h"
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

    if (g_cfg.sparse_mode != SPARSE_DENSE) {
        apply_structured_sparsity_to_matrix_a(ms.a_raw, g_cfg.sparse_mode, prec, rng_state);
        for (int i = 0; i < 8; i++) {
            for (int k = 0; k < 8; k++) {
                switch (prec) {
                    case PREC_FP4_E2M1:
                        ms.a_fp9[i][k] = fp4_to_fp9(ms.a_raw[i][k]);
                        break;
                    case PREC_FP8_E4M3:
                        ms.a_fp9[i][k] = fp8_e4m3_to_fp9(ms.a_raw[i][k]);
                        break;
                    case PREC_FP8_E5M2:
                        ms.a_fp9[i][k] = fp8_e5m2_to_fp9(ms.a_raw[i][k]);
                        break;
                    case PREC_FP16:
                        ms.a_fp9[i][k] = fp16_to_fp9(ms.a_raw[i][k]);
                        break;
                    default:
                        break;
                }
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




const char* sparse_mode_name(SparseMode mode) {
    switch (mode) {
        case SPARSE_DENSE: return "DENSE";
        case SPARSE_2_TO_4: return "SPARSE_2_TO_4";
        case SPARSE_1_TO_4: return "SPARSE_1_TO_4";
        default: return "UNKNOWN";
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
        uint32_t ref_q[8][8];       // 量化后参考（真正应与OUTPUT直接比较）
        double   ref_otc[8][8];     // 现有 reference_matmul 风格参考
        double   ref_fp32[8][8];    // 高精度参考
        int      issue_cycle;
        int      done_cycle;
        double   max_abs_err_q;
        double   max_abs_err_fp32;
    };

    // ===== 当前先做功能调试，先不要一上来跑 8 个 back-to-back =====
    const int num_jobs = 1;

    // ===== 你当前这版 OUTPUT 的数值形态明显是 FP8_E5M2 网格 =====
    // 如果后面你把 DUT 最终输出改成可配置精度，再把这里改成 g_cfg.out_precisions[0]
    const PrecisionType out_prec = PREC_FP8_E5M2;

    auto quantize_double_to_output_bits = [&](double v, PrecisionType p) -> uint32_t {
        switch (p) {
            case PREC_FP8_E4M3:
                return (uint32_t)double_to_fp8_e4m3(v);
            case PREC_FP8_E5M2:
                return (uint32_t)double_to_fp8_e5m2(v);
            case PREC_FP16:
                return (uint32_t)double_to_fp16(v);
            case PREC_FP32: {
                float f = (float)v;
                uint32_t bits = 0;
                std::memcpy(&bits, &f, sizeof(float));
                return bits;
            }
            case PREC_FP4_E2M1:
            default:
                return (uint32_t)double_to_fp4(v);
        }
    };

    auto calc_max_abs_err_bits_vs_double =
        [&](const uint32_t out_bits[8][8], PrecisionType p, const double ref[8][8]) -> double {
            double mx = 0.0;
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    double out_v = output_bits_to_double(out_bits[i][j], p);
                    double err = std::fabs(out_v - ref[i][j]);
                    if (err > mx) mx = err;
                }
            }
            return mx;
        };

    auto calc_max_abs_err_bits_vs_bits =
        [&](const uint32_t out_bits[8][8], PrecisionType p, const uint32_t ref_bits[8][8]) -> double {
            double mx = 0.0;
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    double out_v = output_bits_to_double(out_bits[i][j], p);
                    double ref_v = output_bits_to_double(ref_bits[i][j], p);
                    double err = std::fabs(out_v - ref_v);
                    if (err > mx) mx = err;
                }
            }
            return mx;
        };

    std::vector<JobResult> results;
    results.reserve(num_jobs);

    for (int job = 0; job < num_jobs; job++) {
        MatrixSet ms = generate_random_matrices(g_cfg.precisions[0]);

        JobResult jr = {};
        jr.issue_cycle = -1;
        jr.done_cycle  = -1;
        jr.max_abs_err_q = 0.0;
        jr.max_abs_err_fp32 = 0.0;

        // 1) 现有 OpenTensorCore 风格参考（你原函数已有）
        reference_matmul(ms.a_fp9, ms.b_fp9, ms.c_fp22, jr.ref_otc, g_cfg.rm);

        // 2) 真正的 FP32 累加参考
        golden_fp32_matmul(ms, g_cfg.precisions[0], jr.ref_fp32);

        // 3) 把“现有参考”量化到 DUT 输出精度，作为直接比对对象
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                jr.ref_q[i][j] = quantize_double_to_output_bits(jr.ref_otc[i][j], out_prec);
            }
        }

        results.push_back(jr);

        // ===== 单 job 调试：先发 1 次有效输入，后面全是 invalid 直到出结果 =====
        sim.load_inputs(ms.a_raw, ms.b_raw, ms.c_raw);
        results.back().issue_cycle = sim.cycle_count;

        bool issued = true;
        while (sim.jobs_completed < job + 1) {
            if (issued) {
                sim.load_invalid();
            }

            if (sim.run()) {
                auto& done_jr = results[sim.jobs_completed - 1];
                for (int i = 0; i < 8; i++) {
                    for (int j = 0; j < 8; j++) {
                        done_jr.d_out[i][j] = sim.d_out[i][j];
                    }
                }
                done_jr.done_cycle = sim.cycle_count;
            }
        }

        auto& done_jr = results.back();
        done_jr.max_abs_err_q =
            calc_max_abs_err_bits_vs_bits(done_jr.d_out, out_prec, done_jr.ref_q);
        done_jr.max_abs_err_fp32 =
            calc_max_abs_err_bits_vs_double(done_jr.d_out, out_prec, done_jr.ref_fp32);
    }

    // ===== 打印 =====
    for (int job = 0; job < num_jobs; job++) {
        auto& jr = results[job];

        print_matrix_double("REF_OTC_STYLE", jr.ref_otc);
        print_matrix_double("REF_FP32_ACC",  jr.ref_fp32);
        print_matrix_output("REF_QUANTIZED", jr.ref_q, out_prec);
        print_matrix_output("OUTPUT",        jr.d_out, out_prec);

        printf("    issue_cycle=%d, done_cycle=%d, latency=%d cycles\n",
               jr.issue_cycle, jr.done_cycle,
               (jr.issue_cycle >= 0 && jr.done_cycle >= 0) ? (jr.done_cycle - jr.issue_cycle) : -1);

        printf("    max_abs_err (OUTPUT vs REF_QUANTIZED) = %.8f\n", jr.max_abs_err_q);
        printf("    max_abs_err (OUTPUT vs REF_FP32_ACC)  = %.8f\n", jr.max_abs_err_fp32);
        printf("\n");
    }

    printf("\n  Total: %d cycles for %d jobs | %.1f cycles/matmul\n",
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


SparseMode parse_sparse_mode(const char* s) {
    if (strcmp(s, "0") == 0 || strcmp(s, "DENSE") == 0) return SPARSE_DENSE;
    if (strcmp(s, "1") == 0 || strcmp(s, "2:4") == 0 || strcmp(s, "SPARSE_2_TO_4") == 0) return SPARSE_2_TO_4;
    if (strcmp(s, "2") == 0 || strcmp(s, "1:4") == 0 || strcmp(s, "SPARSE_1_TO_4") == 0) return SPARSE_1_TO_4;
    fprintf(stderr, "  Error: Unknown sparse mode '%s'\n", s);
    fprintf(stderr, "  Valid: 0|DENSE, 1|2:4|SPARSE_2_TO_4, 2|1:4|SPARSE_1_TO_4\n\n");
    exit(1);
}

bool parse_args(int argc, char* argv[]) {
    g_cfg.precisions.clear();
    g_cfg.out_precisions.clear();
    g_cfg.test_id  = 0;
    g_cfg.rm       = RNE;
    g_cfg.seed     = 0;
    g_cfg.sparse_mode = SPARSE_DENSE;
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
        } else if ((strcmp(argv[i], "--sparse-mode") == 0 || strcmp(argv[i], "--sparse") == 0) && i + 1 < argc) {
            g_cfg.sparse_mode = parse_sparse_mode(argv[++i]);
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
