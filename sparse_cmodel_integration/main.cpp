#include "tensor_core_top.h"
#include "config_register.h"
#include "fp_types.h"
#include "sparse_cmodel.h"
#include "fmul_s1.h"
#include "fmul_s2.h"
#include "fmul_s3.h"
#include "fadd_s1.h"
#include "fadd_s2.h"
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


static SparseMode g_sparse_mode_shadow = SPARSE_DENSE;

template <typename T>
auto get_cfg_sparse_mode_impl(const T& cfg, int) -> decltype((void)cfg.sparse_mode, SparseMode()) {
    return static_cast<SparseMode>(cfg.sparse_mode);
}

template <typename T>
SparseMode get_cfg_sparse_mode_impl(const T&, long) {
    return g_sparse_mode_shadow;
}

inline SparseMode current_sparse_mode() {
    return get_cfg_sparse_mode_impl(g_cfg, 0);
}

template <typename T>
auto set_cfg_sparse_mode_impl(T& cfg, SparseMode mode, int) -> decltype((void)(cfg.sparse_mode = mode), void()) {
    cfg.sparse_mode = mode;
    g_sparse_mode_shadow = mode;
}

template <typename T>
void set_cfg_sparse_mode_impl(T&, SparseMode mode, long) {
    g_sparse_mode_shadow = mode;
}

inline void set_current_sparse_mode(SparseMode mode) {
    set_cfg_sparse_mode_impl(g_cfg, mode, 0);
}

inline uint32_t local_convert_fp22_to_out(uint32_t fp22, PrecisionType out_prec, RoundingMode) {
    double v = fp22_to_double(fp22);
    switch (out_prec) {
        case PREC_FP8_E4M3: return (uint32_t)double_to_fp8_e4m3(v);
        case PREC_FP8_E5M2: return (uint32_t)double_to_fp8_e5m2(v);
        case PREC_FP16:     return (uint32_t)double_to_fp16(v);
        case PREC_FP32: {
            float f = (float)v;
            uint32_t bits = 0;
            std::memcpy(&bits, &f, sizeof(float));
            return bits;
        }
        case PREC_FP4_E2M1:
        default:            return (uint32_t)double_to_fp4(v);
    }
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

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            double va = rand_double(-1.0, 1.0);
            double vb = rand_double(-1.0, 1.0);
            double vc = rand_double(-1.0, 1.0);
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
                default:
                    break;
            }
        }
    }

    if (current_sparse_mode() != SPARSE_DENSE) {
        apply_structured_sparsity_to_matrix_a(ms.a_raw, current_sparse_mode(), prec, rng_state);
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
    std::printf("    %s\n", title);
    for (int i = 0; i < 8; i++) {
        std::printf("      ");
        for (int j = 0; j < 8; j++) {
            std::printf("%9.4f ", m[i][j]);
        }
        std::printf("\n");
    }
}

void print_matrix_output(const char* title, const uint32_t m[8][8], PrecisionType out_prec) {
    std::printf("    %s\n", title);
    for (int i = 0; i < 8; i++) {
        std::printf("      ");
        for (int j = 0; j < 8; j++) {
            std::printf("%9.4f ", output_bits_to_double(m[i][j], out_prec));
        }
        std::printf("\n");
    }
}

static uint16_t ref_mul_fp13(uint16_t a_fp9, uint16_t b_fp9, RoundingMode rm) {
    uint32_t a_pad = ((uint32_t)a_fp9) << 4;
    uint32_t b_pad = ((uint32_t)b_fp9) << 4;
    auto s1 = fmul_s1(a_pad, b_pad, 5, 8, rm);
    auto s2 = fmul_s2(a_pad, b_pad, 5, 8, s1);
    return (uint16_t)(fmul_s3(s2, 5, 8) & 0x1FFF);
}

static uint16_t ref_add_fp13(uint16_t a_fp13, uint16_t b_fp13, RoundingMode rm) {
    auto s1 = fadd_s1(a_fp13, b_fp13, 5, 8, 8, rm);
    return (uint16_t)(fadd_s2(s1, 5, 8) & 0x1FFF);
}

static uint32_t ref_add_fp22(uint32_t a_fp22, uint32_t b_fp22, RoundingMode rm) {
    auto s1 = fadd_s1(a_fp22, b_fp22, 8, 14, 14, rm);
    return fadd_s2(s1, 8, 14);
}

static uint32_t ref_one_output_bits(const MatrixSet& ms,
                                    int row,
                                    int col,
                                    PrecisionType in_prec,
                                    PrecisionType out_prec,
                                    RoundingMode rm,
                                    SparseMode sparse_mode,
                                    double* ref_prequant_double = nullptr) {
    uint32_t dense_a_raw[8] = {0};
    uint32_t dense_b_raw[8] = {0};
    for (int k = 0; k < 8; ++k) {
        dense_a_raw[k] = ms.a_raw[row][k];
        dense_b_raw[k] = ms.b_raw[k][col];
    }

    uint32_t routed_a_raw[8] = {0};
    uint32_t routed_b_raw[8] = {0};

    if (sparse_mode != SPARSE_DENSE) {
        uint32_t packed_payload[8] = {0};
        uint8_t packed_meta[2] = {0};
        prepare_sparse_a_payload_and_meta(ms.a_raw[row],
                                          sparse_mode,
                                          in_prec,
                                          packed_payload,
                                          packed_meta);
        input_parser_bypass_expand_sparse_operands(
            packed_payload,
            dense_b_raw,
            packed_meta,
            sparse_mode,
            in_prec,
            routed_a_raw,
            routed_b_raw);
    } else {
        for (int k = 0; k < 8; ++k) {
            routed_a_raw[k] = dense_a_raw[k];
            routed_b_raw[k] = dense_b_raw[k];
        }
    }

    uint16_t prod_fp13[8] = {0};
    for (int k = 0; k < 8; ++k) {
        uint16_t a_fp9 = convert_to_fp9(routed_a_raw[k], in_prec);
        uint16_t b_fp9 = convert_to_fp9(routed_b_raw[k], in_prec);
        prod_fp13[k] = ref_mul_fp13(a_fp9, b_fp9, rm);
    }

    uint16_t l0[4];
    l0[0] = ref_add_fp13(prod_fp13[0], prod_fp13[4], rm);
    l0[1] = ref_add_fp13(prod_fp13[1], prod_fp13[5], rm);
    l0[2] = ref_add_fp13(prod_fp13[2], prod_fp13[6], rm);
    l0[3] = ref_add_fp13(prod_fp13[3], prod_fp13[7], rm);

    uint16_t l1[2];
    l1[0] = ref_add_fp13(l0[0], l0[2], rm);
    l1[1] = ref_add_fp13(l0[1], l0[3], rm);

    uint16_t l2 = ref_add_fp13(l1[0], l1[1], rm);

    uint32_t mac_fp22 = fp13_to_fp22(l2);
    uint32_t c_fp22 = fp16_to_fp22((uint16_t)(ms.c_raw[row][col] & 0xFFFF));
    uint32_t d_fp22 = ref_add_fp22(mac_fp22, c_fp22, rm);

    if (ref_prequant_double) {
        *ref_prequant_double = fp22_to_double(d_fp22);
    }

    return local_convert_fp22_to_out(d_fp22, out_prec, rm);
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
            float c = (float)raw_to_double(ms.c_raw[i][j], PREC_FP16);
            out[i][j] = (double)(acc + c);
        }
    }
}

double calc_max_abs_err_bits_vs_double(const uint32_t out_bits[8][8],
                                       PrecisionType out_prec,
                                       const double ref[8][8]) {
    double mx = 0.0;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            double out_v = output_bits_to_double(out_bits[i][j], out_prec);
            double err = std::fabs(out_v - ref[i][j]);
            if (err > mx) mx = err;
        }
    }
    return mx;
}

double calc_max_abs_err_bits_vs_bits(const uint32_t out_bits[8][8],
                                     PrecisionType out_prec,
                                     const uint32_t ref_bits[8][8]) {
    double mx = 0.0;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            double out_v = output_bits_to_double(out_bits[i][j], out_prec);
            double ref_v = output_bits_to_double(ref_bits[i][j], out_prec);
            double err = std::fabs(out_v - ref_v);
            if (err > mx) mx = err;
        }
    }
    return mx;
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

void test_pipelined_throughput() {
    std::printf("Sparse mode: %s\n", sparse_mode_name(current_sparse_mode()));

    TensorCoreTop sim;
    sim.reset();

    struct JobResult {
        uint32_t d_out[8][8];
        uint32_t ref_q[8][8];
        double ref_otc[8][8];
        double ref_fp32[8][8];
        int issue_cycle;
        int done_cycle;
        double max_abs_err_q;
        double max_abs_err_fp32;
    };

    const int num_jobs = 8;
    const PrecisionType out_prec = g_cfg.out_precisions.empty()
        ? PREC_FP8_E5M2
        : g_cfg.out_precisions.at(0);

    std::vector<JobResult> results;
    results.reserve(num_jobs);

    int jobs_issued = 0;
    while (sim.jobs_completed < num_jobs) {
        if (jobs_issued < num_jobs && sim.in_ready(true)) {
            MatrixSet ms = generate_random_matrices(g_cfg.precisions[0]);

            JobResult jr = {};
            jr.issue_cycle = sim.cycle_count;
            jr.done_cycle = -1;
            jr.max_abs_err_q = 0.0;
            jr.max_abs_err_fp32 = 0.0;

            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    double preq = 0.0;
                    jr.ref_q[i][j] = ref_one_output_bits(ms,
                                                         i,
                                                         j,
                                                         g_cfg.precisions[0],
                                                         out_prec,
                                                         g_cfg.rm,
                                                         current_sparse_mode(),
                                                         &preq);
                    jr.ref_otc[i][j] = preq;
                }
            }

            golden_fp32_matmul(ms, g_cfg.precisions[0], jr.ref_fp32);
            results.push_back(jr);
            sim.load_inputs(ms.a_raw, ms.b_raw, ms.c_raw);
            ++jobs_issued;
        } else {
            sim.load_invalid();
        }

        if (sim.run()) {
            JobResult& done_jr = results[sim.jobs_completed - 1];
            for (int i = 0; i < 8; ++i) {
                for (int j = 0; j < 8; ++j) {
                    done_jr.d_out[i][j] = sim.d_out[i][j];
                }
            }
            done_jr.done_cycle = sim.cycle_count;
            done_jr.max_abs_err_q = calc_max_abs_err_bits_vs_bits(done_jr.d_out, out_prec, done_jr.ref_q);
            done_jr.max_abs_err_fp32 = calc_max_abs_err_bits_vs_double(done_jr.d_out, out_prec, done_jr.ref_fp32);
        }
    }

    double worst_err_q = 0.0;
    double worst_err_fp32 = 0.0;
    int first_fail_job = -1;
    for (int job = 0; job < num_jobs; ++job) {
        JobResult& jr = results[job];
        if (jr.max_abs_err_q > worst_err_q) worst_err_q = jr.max_abs_err_q;
        if (jr.max_abs_err_fp32 > worst_err_fp32) worst_err_fp32 = jr.max_abs_err_fp32;
        if (first_fail_job < 0 && jr.max_abs_err_q != 0.0) first_fail_job = job;

        std::printf("    job=%d issue_cycle=%d done_cycle=%d latency=%d cycles\n",
                    job,
                    jr.issue_cycle,
                    jr.done_cycle,
                    (jr.issue_cycle >= 0 && jr.done_cycle >= 0) ? (jr.done_cycle - jr.issue_cycle) : -1);
        std::printf("      max_abs_err (OUTPUT vs REF_QUANTIZED) = %.8f\n", jr.max_abs_err_q);
        std::printf("      max_abs_err (OUTPUT vs REF_FP32_ACC)  = %.8f\n", jr.max_abs_err_fp32);
    }

    if (first_fail_job >= 0) {
        JobResult& jr = results[first_fail_job];
        std::printf("\n  First failing job = %d\n", first_fail_job);
        print_matrix_double("REF_OTC_STYLE", jr.ref_otc);
        print_matrix_double("REF_FP32_ACC", jr.ref_fp32);
        print_matrix_output("REF_QUANTIZED", jr.ref_q, out_prec);
        print_matrix_output("OUTPUT", jr.d_out, out_prec);
    }

    std::printf("\n  Summary: worst_err_q=%.8f, worst_err_fp32=%.8f\n",
                worst_err_q, worst_err_fp32);
    std::printf("  Total: %d cycles for %d jobs | %.1f cycles/matmul\n",
                sim.cycle_count, num_jobs, sim.cycle_count / (double)num_jobs);
}

PrecisionType parse_precision(const char* s) {
    if (std::strcmp(s, "FP4_E2M1") == 0 || std::strcmp(s, "FP4") == 0)  return PREC_FP4_E2M1;
    if (std::strcmp(s, "FP8_E4M3") == 0 || std::strcmp(s, "E4M3") == 0) return PREC_FP8_E4M3;
    if (std::strcmp(s, "FP8_E5M2") == 0 || std::strcmp(s, "E5M2") == 0) return PREC_FP8_E5M2;
    if (std::strcmp(s, "FP16") == 0) return PREC_FP16;
    if (std::strcmp(s, "FP32") == 0) return PREC_FP32;
    std::fprintf(stderr, "  Error: Unknown precision '%s'\n", s);
    std::fprintf(stderr, "  Valid: FP4_E2M1 | FP8_E4M3 | FP8_E5M2 | FP16 | FP32\n\n");
    std::exit(1);
}

RoundingMode parse_rounding(const char* s) {
    if (std::strcmp(s, "RNE") == 0) return RNE;
    if (std::strcmp(s, "RTZ") == 0) return RTZ;
    if (std::strcmp(s, "RDN") == 0) return RDN;
    if (std::strcmp(s, "RUP") == 0) return RUP;
    if (std::strcmp(s, "RMM") == 0) return RMM;
    std::fprintf(stderr, "  Error: Unknown rounding mode '%s'\n", s);
    std::fprintf(stderr, "  Valid: RNE | RTZ | RDN | RUP | RMM\n\n");
    std::exit(1);
}

SparseMode parse_sparse_mode(const char* s) {
    if (std::strcmp(s, "0") == 0 || std::strcmp(s, "DENSE") == 0) return SPARSE_DENSE;
    if (std::strcmp(s, "1") == 0 || std::strcmp(s, "2:4") == 0 || std::strcmp(s, "SPARSE_2_TO_4") == 0) return SPARSE_2_TO_4;
    if (std::strcmp(s, "2") == 0 || std::strcmp(s, "1:4") == 0 || std::strcmp(s, "SPARSE_1_TO_4") == 0) return SPARSE_1_TO_4;
    std::fprintf(stderr, "  Error: Unknown sparse mode '%s'\n", s);
    std::fprintf(stderr, "  Valid: 0|DENSE, 1|2:4|SPARSE_2_TO_4, 2|1:4|SPARSE_1_TO_4\n\n");
    std::exit(1);
}

bool parse_args(int argc, char* argv[]) {
    g_cfg.precisions.clear();
    g_cfg.out_precisions.clear();
    g_cfg.test_id = 0;
    g_cfg.rm = RNE;
    g_cfg.seed = 0;
    set_current_sparse_mode(SPARSE_DENSE);
    g_cfg.show_help = false;

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            g_cfg.show_help = true;
            return true;
        } else if (std::strcmp(argv[i], "--prec") == 0 && i + 1 < argc) {
            g_cfg.precisions.push_back(parse_precision(argv[++i]));
        } else if (std::strcmp(argv[i], "--test") == 0 && i + 1 < argc) {
            g_cfg.test_id = std::atoi(argv[++i]);
        } else if ((std::strcmp(argv[i], "--out-prec") == 0 || std::strcmp(argv[i], "--output-prec") == 0) && i + 1 < argc) {
            g_cfg.out_precisions.push_back(parse_precision(argv[++i]));
        } else if (std::strcmp(argv[i], "--rm") == 0 && i + 1 < argc) {
            g_cfg.rm = parse_rounding(argv[++i]);
        } else if ((std::strcmp(argv[i], "--sparse-mode") == 0 || std::strcmp(argv[i], "--sparse") == 0) && i + 1 < argc) {
            set_current_sparse_mode(parse_sparse_mode(argv[++i]));
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            g_cfg.seed = (uint32_t)std::atol(argv[++i]);
        } else {
            std::fprintf(stderr, "  Error: Unknown argument '%s'\n\n", argv[i]);
            return false;
        }
    }

    if (g_cfg.precisions.empty()) {
        g_cfg.precisions = {PREC_FP4_E2M1, PREC_FP8_E4M3, PREC_FP8_E5M2, PREC_FP16};
    }
    if (g_cfg.out_precisions.empty()) {
        g_cfg.out_precisions = {PREC_FP8_E5M2};
    }

    return true;
}

static void print_summary() {}

int main(int argc, char* argv[]) {
    if (!parse_args(argc, argv)) {
        return 1;
    }

    rng_state = g_cfg.seed ? g_cfg.seed : (uint32_t)std::time(nullptr);

    bool run_all = (g_cfg.test_id == 0);
    if (run_all) print_summary();
    if (run_all || g_cfg.test_id == 2) test_pipelined_throughput();

    return 0;
}

