#include "tensor_core_top.h"
#include "config_register.h"
#include "fp_types.h"
#include "fp22_to_fp16.h"
#include "sparse_select.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kMacroM = 16;
constexpr uint32_t kMacroN = 16;
constexpr uint32_t kPrimitiveDim = 8;
constexpr uint32_t kSparseNone = 0;
constexpr uint32_t kSparse2To4 = 1;
constexpr uint32_t kSparse1To4 = 2;

using PrimitiveFp9 = std::array<std::array<uint16_t, kPrimitiveDim>, kPrimitiveDim>;
using PrimitiveFp22 = std::array<std::array<uint32_t, kPrimitiveDim>, kPrimitiveDim>;
using SparseRowMeta = std::array<uint16_t, kPrimitiveDim>;

static uint32_t rng_state = 1;

struct QuantMatrix {
  uint32_t rows = 0;
  uint32_t cols = 0;
  PrecisionType prec = PREC_FP16;
  std::vector<uint32_t> raw;

  QuantMatrix() = default;

  QuantMatrix(uint32_t rows_, uint32_t cols_, PrecisionType prec_)
      : rows(rows_), cols(cols_), prec(prec_), raw(rows_ * cols_, 0) {}

  uint32_t& at(uint32_t row, uint32_t col) {
    return raw.at(row * cols + col);
  }

  uint32_t at(uint32_t row, uint32_t col) const {
    return raw.at(row * cols + col);
  }
};

struct TestCaseDef {
  int id;
  const char* name;
  uint32_t m;
  uint32_t n;
  uint32_t k;
  PrecisionType a_prec;
  PrecisionType b_prec;
  PrecisionType c_prec;
  PrecisionType d_prec;
  uint32_t sparse_mode;
  bool include_c;
  uint32_t macro_k_tile;
  double threshold;
};

struct ErrorStats {
  double max_abs = 0.0;
  double mean_abs = 0.0;
  double rms = 0.0;
  uint32_t mismatches = 0;
};

static constexpr TestCaseDef kTests[] = {
    {1, "dense_mma_fp8_fp8_to_fp16", 16, 16, 16, PREC_FP8_E4M3, PREC_FP8_E4M3,
     PREC_FP16, PREC_FP16, kSparseNone, true, 16, 0.01},
    {2, "asym_mma_fp16_fp8_to_fp32", 16, 16, 16, PREC_FP16, PREC_FP8_E4M3,
     PREC_FP32, PREC_FP32, kSparseNone, true, 16, 0.01},
    {3, "multi_ktile_accum_fp8_fp8_to_fp16", 16, 16, 64, PREC_FP8_E4M3, PREC_FP8_E4M3,
     PREC_FP16, PREC_FP16, kSparseNone, true, 32, 0.03},
    {4, "sparse_2_4_mma_fp8_fp8_to_fp16", 16, 16, 32, PREC_FP8_E4M3, PREC_FP8_E4M3,
     PREC_FP16, PREC_FP16, kSparse2To4, true, 32, 0.05},
    {5, "sparse_1_4_mma_fp8_fp8_to_fp16", 16, 16, 64, PREC_FP8_E4M3, PREC_FP8_E4M3,
     PREC_FP16, PREC_FP16, kSparse1To4, true, 32, 0.05},
};

uint32_t xorshift32() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}

double rand_double(double lo, double hi) {
  double t = static_cast<double>(xorshift32() & 0xffffu) / 65535.0;
  return lo + (hi - lo) * t;
}

const char* prec_name(PrecisionType p) {
  switch (p) {
  case PREC_FP4_E2M1: return "FP4_E2M1";
  case PREC_FP8_E4M3: return "FP8_E4M3";
  case PREC_FP8_E5M2: return "FP8_E5M2";
  case PREC_FP9:      return "FP9";
  case PREC_FP16:     return "FP16";
  case PREC_FP32:     return "FP32";
  default:            return "UNKNOWN";
  }
}

const char* sparse_name(uint32_t sparse_mode) {
  switch (sparse_mode) {
  case kSparseNone: return "dense";
  case kSparse2To4: return "2:4";
  case kSparse1To4: return "1:4";
  default:          return "unknown";
  }
}

const char* rm_name(RoundingMode rm) {
  switch (rm) {
  case RNE: return "RNE";
  case RTZ: return "RTZ";
  case RDN: return "RDN";
  case RUP: return "RUP";
  case RMM: return "RMM";
  default:  return "UNKNOWN";
  }
}

uint32_t encode_fp32(double value) {
  float f = static_cast<float>(value);
  uint32_t bits = 0;
  std::memcpy(&bits, &f, sizeof(bits));
  return bits;
}

uint32_t encode_raw(double value, PrecisionType prec) {
  switch (prec) {
  case PREC_FP4_E2M1: return double_to_fp4(value);
  case PREC_FP8_E4M3: return double_to_fp8_e4m3(value);
  case PREC_FP8_E5M2: return double_to_fp8_e5m2(value);
  case PREC_FP16:     return double_to_fp16(value);
  case PREC_FP32:     return encode_fp32(value);
  default:
    std::fprintf(stderr, "encode_raw: unsupported precision %s\n", prec_name(prec));
    std::exit(1);
  }
}

double decode_raw(uint32_t bits, PrecisionType prec) {
  switch (prec) {
  case PREC_FP4_E2M1: return fp4_to_double(static_cast<uint8_t>(bits & 0xfu));
  case PREC_FP8_E4M3: return fp8_e4m3_to_double(static_cast<uint8_t>(bits & 0xffu));
  case PREC_FP8_E5M2: return fp8_e5m2_to_double(static_cast<uint8_t>(bits & 0xffu));
  case PREC_FP16:     return fp16_to_double(static_cast<uint16_t>(bits & 0xffffu));
  case PREC_FP32: {
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return static_cast<double>(f);
  }
  default:
    std::fprintf(stderr, "decode_raw: unsupported precision %s\n", prec_name(prec));
    std::exit(1);
  }
}

double fp22_to_output_double(uint32_t fp22, PrecisionType out_prec) {
  switch (out_prec) {
  case PREC_FP8_E4M3:
    return fp8_e4m3_to_double(fp22_to_fp8_e4m3(fp22, g_cfg.rm));
  case PREC_FP8_E5M2:
    return fp8_e5m2_to_double(fp22_to_fp8_e5m2(fp22, g_cfg.rm));
  case PREC_FP16:
    return fp16_to_double(fp22_to_fp16(fp22));
  case PREC_FP32: {
    uint32_t raw = fp22_to_fp32(fp22);
    float f = 0.0f;
    std::memcpy(&f, &raw, sizeof(f));
    return static_cast<double>(f);
  }
  default:
    std::fprintf(stderr, "fp22_to_output_double: unsupported precision %s\n", prec_name(out_prec));
    std::exit(1);
  }
}

uint32_t sample_quantized_raw(PrecisionType prec,
                              double lo,
                              double hi,
                              bool require_nonzero) {
  for (uint32_t attempt = 0; attempt < 128; ++attempt) {
    double candidate = rand_double(lo, hi);
    uint32_t raw = encode_raw(candidate, prec);
    if (!require_nonzero || decode_raw(raw, prec) != 0.0) {
      return raw;
    }
  }

  double fallback = (hi > 0.0) ? hi : 1.0;
  if (require_nonzero) {
    return encode_raw(fallback, prec);
  }
  return encode_raw(0.0, prec);
}

void fill_dense_matrix(QuantMatrix* matrix, double lo, double hi) {
  if (matrix == nullptr) {
    return;
  }
  for (uint32_t row = 0; row < matrix->rows; ++row) {
    for (uint32_t col = 0; col < matrix->cols; ++col) {
      matrix->at(row, col) = sample_quantized_raw(matrix->prec, lo, hi, false);
    }
  }
}

void fill_zero_matrix(QuantMatrix* matrix) {
  if (matrix == nullptr) {
    return;
  }
  std::fill(matrix->raw.begin(), matrix->raw.end(), 0u);
}

void fill_identity_a(QuantMatrix* matrix, uint32_t k_base) {
  if (matrix == nullptr) {
    return;
  }
  for (uint32_t row = 0; row < matrix->rows; ++row) {
    uint32_t col = k_base + row;
    if (col < matrix->cols) {
      matrix->at(row, col) = encode_raw(1.0, matrix->prec);
    }
  }
}

void fill_sparse_matrix_2_4(QuantMatrix* matrix, double lo, double hi) {
  if (matrix == nullptr) {
    return;
  }
  if ((matrix->cols % 8u) != 0) {
    std::fprintf(stderr, "fill_sparse_matrix_2_4: K=%u must be divisible by 8\n", matrix->cols);
    std::exit(1);
  }
  fill_zero_matrix(matrix);
  for (uint32_t row = 0; row < matrix->rows; ++row) {
    for (uint32_t slice = 0; slice < matrix->cols; slice += 8) {
      for (uint32_t group = 0; group < 2; ++group) {
        bool chosen[4] = {false, false, false, false};
        uint32_t picks = 0;
        while (picks < 2) {
          uint32_t lane = xorshift32() & 0x3u;
          if (!chosen[lane]) {
            chosen[lane] = true;
            ++picks;
          }
        }
        for (uint32_t lane = 0; lane < 4; ++lane) {
          uint32_t col = slice + group * 4 + lane;
          matrix->at(row, col) = chosen[lane]
                               ? sample_quantized_raw(matrix->prec, lo, hi, true)
                               : 0u;
        }
      }
    }
  }
}

void fill_sparse_matrix_1_4(QuantMatrix* matrix, double lo, double hi) {
  if (matrix == nullptr) {
    return;
  }
  if ((matrix->cols % 8u) != 0) {
    std::fprintf(stderr, "fill_sparse_matrix_1_4: K=%u must be divisible by 8\n", matrix->cols);
    std::exit(1);
  }
  fill_zero_matrix(matrix);
  for (uint32_t row = 0; row < matrix->rows; ++row) {
    for (uint32_t slice = 0; slice < matrix->cols; slice += 8) {
      for (uint32_t group = 0; group < 2; ++group) {
        uint32_t pick = xorshift32() & 0x3u;
        for (uint32_t lane = 0; lane < 4; ++lane) {
          uint32_t col = slice + group * 4 + lane;
          matrix->at(row, col) = (lane == pick)
                               ? sample_quantized_raw(matrix->prec, lo, hi, true)
                               : 0u;
        }
      }
    }
  }
}

void init_case_inputs(const TestCaseDef& tc,
                      QuantMatrix* a,
                      QuantMatrix* b,
                      QuantMatrix* c) {
  if (a == nullptr || b == nullptr || c == nullptr) {
    return;
  }

  *a = QuantMatrix(tc.m, tc.k, tc.a_prec);
  *b = QuantMatrix(tc.k, tc.n, tc.b_prec);
  *c = QuantMatrix(tc.m, tc.n, tc.c_prec);

  if (tc.id == 1 || tc.id == 2) {
    fill_zero_matrix(a);
    fill_identity_a(a, 0);
    fill_dense_matrix(b, 0.0, 0.25);
  } else if (tc.id == 3) {
    fill_zero_matrix(a);
    fill_identity_a(a, 0);
    fill_identity_a(a, 32);
    fill_dense_matrix(b, 0.0, 0.25);
  } else if (tc.sparse_mode == kSparse2To4) {
    fill_sparse_matrix_2_4(a, 0.0, 0.25);
    fill_dense_matrix(b, 0.0, 0.25);
  } else if (tc.sparse_mode == kSparse1To4) {
    fill_sparse_matrix_1_4(a, 0.0, 0.25);
    fill_dense_matrix(b, 0.0, 0.25);
  } else {
    fill_dense_matrix(a, -0.25, 0.25);
    fill_dense_matrix(b, -0.25, 0.25);
  }

  if (tc.include_c) {
    fill_dense_matrix(c, 0.0, 0.25);
  } else {
    fill_zero_matrix(c);
  }
}

PrimitiveFp22 initial_c_subtile(const QuantMatrix& c,
                                uint32_t m_base,
                                uint32_t n_base) {
  PrimitiveFp22 out = {};
  for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
    for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
      out[i][j] = convert_c_to_fp22(c.at(m_base + i, n_base + j), c.prec);
    }
  }
  return out;
}

PrimitiveFp9 dense_a_primitive(const QuantMatrix& a,
                               uint32_t m_base,
                               uint32_t k_base) {
  PrimitiveFp9 out = {};
  for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
    for (uint32_t k = 0; k < kPrimitiveDim; ++k) {
      out[i][k] = convert_to_fp9(a.at(m_base + i, k_base + k), a.prec);
    }
  }
  return out;
}

PrimitiveFp9 dense_b_primitive(const QuantMatrix& b,
                               uint32_t k_base,
                               uint32_t n_base) {
  PrimitiveFp9 out = {};
  for (uint32_t k = 0; k < kPrimitiveDim; ++k) {
    for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
      out[k][j] = convert_to_fp9(b.at(k_base + k, n_base + j), b.prec);
    }
  }
  return out;
}

void build_sparse_compact_primitive(const QuantMatrix& a,
                                    const QuantMatrix& b,
                                    uint32_t sparse_mode,
                                    uint32_t m_base,
                                    uint32_t n_base,
                                    uint32_t k_base,
                                    PrimitiveFp9* a_compact_out,
                                    PrimitiveFp9* b_dense_out,
                                    SparseRowMeta* row_meta_out) {
  if (a_compact_out == nullptr || b_dense_out == nullptr || row_meta_out == nullptr) {
    return;
  }

  PrimitiveFp9 a_compact = {};
  PrimitiveFp9 b_dense = dense_b_primitive(b, k_base, n_base);
  SparseRowMeta row_meta_by_row = {};

  for (uint32_t row = 0; row < kPrimitiveDim; ++row) {
    uint16_t row_meta = 0;
    uint32_t cursor = 0;
    for (uint32_t group = 0; group < 2; ++group) {
      uint32_t base = group * 4;
      std::array<uint32_t, 2> picks = {0, 0};
      uint32_t pick_count = 0;
      for (uint32_t lane = 0; lane < 4; ++lane) {
        uint32_t col = k_base + base + lane;
        if (a.at(m_base + row, col) != 0u) {
          if (pick_count >= picks.size()) {
            std::fprintf(stderr,
                         "Sparse row has too many nonzeros: mode=%s row=%u k_base=%u group=%u\n",
                         sparse_name(sparse_mode), row, k_base, group);
            std::exit(1);
          }
          picks[pick_count++] = lane;
        }
      }

      uint32_t expected = (sparse_mode == kSparse2To4) ? 2u : 1u;
      if (pick_count != expected) {
        std::fprintf(stderr,
                     "Sparse row has wrong nonzero count: mode=%s row=%u k_base=%u group=%u expected=%u got=%u\n",
                     sparse_name(sparse_mode), row, k_base, group, expected, pick_count);
        std::exit(1);
      }

      for (uint32_t idx = 0; idx < expected; ++idx) {
        uint32_t lane = picks[idx];
        uint32_t col = k_base + base + lane;
        a_compact[row][cursor++] = convert_to_fp9(a.at(m_base + row, col), a.prec);
      }

      if (sparse_mode == kSparse2To4) {
        row_meta |= static_cast<uint16_t>(picks[0] & 0x3u) << (group * 4 + 0);
        row_meta |= static_cast<uint16_t>(picks[1] & 0x3u) << (group * 4 + 2);
      } else {
        row_meta |= static_cast<uint16_t>(picks[0] & 0x3u) << (group * 2);
      }
    }

    row_meta_by_row[row] = row_meta;
  }

  *a_compact_out = a_compact;
  *b_dense_out = b_dense;
  *row_meta_out = row_meta_by_row;
}

uint32_t reference_mul_fp9_to_fp22(uint16_t a_fp9, uint16_t b_fp9) {
  auto s1 = fmul_s1(a_fp9, b_fp9, 8, 14, g_cfg.rm);
  auto s2 = fmul_s2(a_fp9, b_fp9, 8, 14, s1);
  return fmul_s3(s2, 8, 14);
}

PrimitiveFp22 reference_primitive(const PrimitiveFp9& a,
                                  const PrimitiveFp9& b,
                                  const PrimitiveFp22& c,
                                  uint32_t sparse_mode,
                                  const SparseRowMeta& row_meta,
                                  uint32_t accum_phase) {
  PrimitiveFp22 out = {};

  for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
    for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
      uint32_t a_src[kPrimitiveDim] = {};
      uint32_t b_src[kPrimitiveDim] = {};
      for (uint32_t k = 0; k < kPrimitiveDim; ++k) {
        a_src[k] = a[i][k];
        b_src[k] = b[k][j];
      }

      uint32_t a4[4] = {};
      uint32_t b4[4] = {};
      if (sparse_mode == kSparseNone) {
        uint32_t base = (accum_phase & 0x1u) * 4u;
        for (uint32_t lane = 0; lane < 4; ++lane) {
          a4[lane] = a_src[base + lane];
          b4[lane] = b_src[base + lane];
        }
      } else if (sparse_mode == kSparse2To4) {
        vortex::sparse::select_sparse_2_4_lane(row_meta[i], a_src, b_src, a4, b4);
      } else if (sparse_mode == kSparse1To4) {
        vortex::sparse::select_sparse_1_4_lane(row_meta[i], a_src, b_src, a4, b4);
      } else {
        std::fprintf(stderr, "reference_primitive: unsupported sparse mode %u\n", sparse_mode);
        std::exit(1);
      }

      uint32_t products[4] = {};
      for (uint32_t lane = 0; lane < 4; ++lane) {
        products[lane] = reference_mul_fp9_to_fp22(static_cast<uint16_t>(a4[lane]),
                                                   static_cast<uint16_t>(b4[lane]));
      }
      uint32_t dot = add4_fp22_rtl::add4_reference(products[0],
                                                   products[1],
                                                   products[2],
                                                   products[3],
                                                   g_cfg.rm);
      out[i][j] = fp22_add_bits(dot, c[i][j], g_cfg.rm);
    }
  }

  return out;
}

PrimitiveFp22 run_primitive(TensorCoreTop* sim,
                            const PrimitiveFp9& a,
                            const PrimitiveFp9& b,
                            const PrimitiveFp22& c,
                            uint32_t sparse_mode,
                            const SparseRowMeta& row_meta,
                            uint32_t accum_phase,
                            uint32_t* elapsed_cycles) {
  if (sim == nullptr) {
    std::fprintf(stderr, "run_primitive: null simulator\n");
    std::exit(1);
  }
  sim->reset();

  uint16_t a_in[kPrimitiveDim][kPrimitiveDim] = {};
  uint16_t b_in[kPrimitiveDim][kPrimitiveDim] = {};
  uint32_t c_in[kPrimitiveDim][kPrimitiveDim] = {};

  for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
    for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
      a_in[i][j] = a[i][j];
      b_in[i][j] = b[i][j];
      c_in[i][j] = c[i][j];
    }
  }

  TensorCoreMeta meta{};
  meta.valid = true;
  meta.accum_phase_id = accum_phase;

  uint16_t sparse_row_meta[kPrimitiveDim] = {};
  for (uint32_t row = 0; row < kPrimitiveDim; ++row) {
    sparse_row_meta[row] = row_meta[row];
  }

  sim->push_uop(a_in, b_in, c_in, meta, sparse_mode, sparse_row_meta);

  TensorCoreRetire retired{};
  for (uint32_t guard = 0; guard < 1024; ++guard) {
    sim->tick(true);
    if (sim->pop_retired(&retired)) {
      PrimitiveFp22 out = {};
      for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
        for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
          out[i][j] = retired.fp22_out[i][j];
        }
      }
      sim->tick(true);
      if (elapsed_cycles != nullptr) {
        *elapsed_cycles += static_cast<uint32_t>(sim->cycle_count);
      }
      return out;
    }
  }

  std::fprintf(stderr, "Primitive did not retire within guard window\n");
  std::exit(1);
}

void accumulate_fp22_subtile(PrimitiveFp22* accum,
                             const PrimitiveFp22& phase_out,
                             bool first_phase) {
  if (accum == nullptr) {
    return;
  }

  if (first_phase) {
    *accum = phase_out;
    return;
  }

  for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
    for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
      (*accum)[i][j] = fp22_add_bits((*accum)[i][j], phase_out[i][j], g_cfg.rm);
    }
  }
}

std::vector<double> build_golden(const TestCaseDef& tc,
                                 const QuantMatrix& a,
                                 const QuantMatrix& b,
                                 const QuantMatrix& c) {
  std::vector<double> golden(tc.m * tc.n, 0.0);

  for (uint32_t m_base = 0; m_base < tc.m; m_base += kPrimitiveDim) {
    for (uint32_t n_base = 0; n_base < tc.n; n_base += kPrimitiveDim) {
      PrimitiveFp22 partial = {};
      bool first_phase = true;

      for (uint32_t macro_k = 0; macro_k < tc.k; macro_k += tc.macro_k_tile) {
        uint32_t macro_k_end = std::min<uint32_t>(macro_k + tc.macro_k_tile, tc.k);
        for (uint32_t k_base = macro_k; k_base < macro_k_end; k_base += kPrimitiveDim) {
          if (tc.sparse_mode == kSparseNone) {
            PrimitiveFp9 a_prim = dense_a_primitive(a, m_base, k_base);
            PrimitiveFp9 b_prim = dense_b_primitive(b, k_base, n_base);
            SparseRowMeta row_meta = {};

            for (uint32_t half = 0; half < 2; ++half) {
              PrimitiveFp22 c_input = first_phase ? initial_c_subtile(c, m_base, n_base)
                                                  : PrimitiveFp22{};
              uint32_t accum_phase = (((k_base / kPrimitiveDim) & 0x1u) * 2u) + half;
              PrimitiveFp22 phase_out = reference_primitive(a_prim,
                                                            b_prim,
                                                            c_input,
                                                            tc.sparse_mode,
                                                            row_meta,
                                                            accum_phase);
              accumulate_fp22_subtile(&partial, phase_out, first_phase);
              first_phase = false;
            }
          } else {
            PrimitiveFp9 a_prim = {};
            PrimitiveFp9 b_prim = {};
            SparseRowMeta row_meta = {};
            build_sparse_compact_primitive(a, b, tc.sparse_mode, m_base, n_base, k_base,
                                           &a_prim, &b_prim, &row_meta);

            PrimitiveFp22 c_input = first_phase ? initial_c_subtile(c, m_base, n_base)
                                                : PrimitiveFp22{};
            uint32_t accum_phase = (k_base / kPrimitiveDim) & 0x1u;
            PrimitiveFp22 phase_out = reference_primitive(a_prim,
                                                          b_prim,
                                                          c_input,
                                                          tc.sparse_mode,
                                                          row_meta,
                                                          accum_phase);
            accumulate_fp22_subtile(&partial, phase_out, first_phase);
            first_phase = false;
          }
        }
      }

      for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
        for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
          golden[(m_base + i) * tc.n + (n_base + j)] =
              fp22_to_output_double(partial[i][j], tc.d_prec);
        }
      }
    }
  }

  return golden;
}

std::vector<double> run_macro_case(const TestCaseDef& tc,
                                   const QuantMatrix& a,
                                   const QuantMatrix& b,
                                   const QuantMatrix& c,
                                   uint32_t* total_cycles) {
  TensorCoreTop sim;
  sim.reset();

  std::vector<double> actual(tc.m * tc.n, 0.0);
  uint32_t cycles = 0;

  for (uint32_t m_base = 0; m_base < tc.m; m_base += kPrimitiveDim) {
    for (uint32_t n_base = 0; n_base < tc.n; n_base += kPrimitiveDim) {
      PrimitiveFp22 partial = {};
      bool first_phase = true;

      for (uint32_t macro_k = 0; macro_k < tc.k; macro_k += tc.macro_k_tile) {
        uint32_t macro_k_end = std::min<uint32_t>(macro_k + tc.macro_k_tile, tc.k);
        for (uint32_t k_base = macro_k; k_base < macro_k_end; k_base += kPrimitiveDim) {
          if (tc.sparse_mode == kSparseNone) {
            PrimitiveFp9 a_prim = dense_a_primitive(a, m_base, k_base);
            PrimitiveFp9 b_prim = dense_b_primitive(b, k_base, n_base);
            SparseRowMeta row_meta = {};

            for (uint32_t half = 0; half < 2; ++half) {
              PrimitiveFp22 c_input = first_phase ? initial_c_subtile(c, m_base, n_base)
                                                  : PrimitiveFp22{};
              uint32_t accum_phase = (((k_base / kPrimitiveDim) & 0x1u) * 2u) + half;
              PrimitiveFp22 phase_out = run_primitive(&sim,
                                                      a_prim,
                                                      b_prim,
                                                      c_input,
                                                      tc.sparse_mode,
                                                      row_meta,
                                                      accum_phase,
                                                      &cycles);
              accumulate_fp22_subtile(&partial, phase_out, first_phase);
              first_phase = false;
            }
          } else {
            PrimitiveFp9 a_prim = {};
            PrimitiveFp9 b_prim = {};
            SparseRowMeta row_meta = {};
            build_sparse_compact_primitive(a, b, tc.sparse_mode, m_base, n_base, k_base,
                                           &a_prim, &b_prim, &row_meta);

            PrimitiveFp22 c_input = first_phase ? initial_c_subtile(c, m_base, n_base)
                                                : PrimitiveFp22{};
            uint32_t accum_phase = (k_base / kPrimitiveDim) & 0x1u;
            PrimitiveFp22 phase_out = run_primitive(&sim,
                                                    a_prim,
                                                    b_prim,
                                                    c_input,
                                                    tc.sparse_mode,
                                                    row_meta,
                                                    accum_phase,
                                                    &cycles);
            accumulate_fp22_subtile(&partial, phase_out, first_phase);
            first_phase = false;
          }
        }
      }

      for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
        for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
          actual[(m_base + i) * tc.n + (n_base + j)] =
              fp22_to_output_double(partial[i][j], tc.d_prec);
        }
      }
    }
  }

  if (total_cycles != nullptr) {
    *total_cycles = cycles;
  }
  return actual;
}

ErrorStats compare_outputs(const std::vector<double>& actual,
                           const std::vector<double>& golden,
                           double threshold) {
  ErrorStats stats;
  if (actual.size() != golden.size()) {
    std::fprintf(stderr, "compare_outputs: size mismatch actual=%zu golden=%zu\n",
                 actual.size(), golden.size());
    std::exit(1);
  }

  double sum_abs = 0.0;
  double sum_sq = 0.0;
  for (size_t idx = 0; idx < actual.size(); ++idx) {
    double diff = std::fabs(actual[idx] - golden[idx]);
    sum_abs += diff;
    sum_sq += diff * diff;
    stats.max_abs = std::max(stats.max_abs, diff);
    if (diff > threshold) {
      ++stats.mismatches;
    }
  }

  if (!actual.empty()) {
    stats.mean_abs = sum_abs / static_cast<double>(actual.size());
    stats.rms = std::sqrt(sum_sq / static_cast<double>(actual.size()));
  }
  return stats;
}

const TestCaseDef* find_test(int id) {
  for (const auto& test : kTests) {
    if (test.id == id) {
      return &test;
    }
  }
  return nullptr;
}

void print_usage(const char* argv0) {
  std::printf("Usage: %s [--test ID] [--seed N] [--rm MODE]\n", argv0);
  std::printf("  --test 0 runs all cases\n");
  std::printf("  --test 1 dense 16x16x16 fp8*fp8->fp16 (+C)\n");
  std::printf("  --test 2 asym  16x16x16 fp16*fp8->fp32 (+C)\n");
  std::printf("  --test 3 multi-K 16x16x64 fp8*fp8->fp16 (+C)\n");
  std::printf("  --test 4 sparse 2:4 16x16x32 fp8*fp8->fp16 (+C)\n");
  std::printf("  --test 5 sparse 1:4 16x16x64 fp8*fp8->fp16 (+C)\n");
  std::printf("  --rm   RNE | RTZ | RDN | RUP | RMM\n");
}

RoundingMode parse_rounding(const char* s) {
  if (std::strcmp(s, "RNE") == 0) return RNE;
  if (std::strcmp(s, "RTZ") == 0) return RTZ;
  if (std::strcmp(s, "RDN") == 0) return RDN;
  if (std::strcmp(s, "RUP") == 0) return RUP;
  if (std::strcmp(s, "RMM") == 0) return RMM;
  std::fprintf(stderr, "Unknown rounding mode '%s'\n", s);
  std::exit(1);
}

bool parse_args(int argc, char* argv[]) {
  g_cfg.test_id = 0;
  g_cfg.rm = RNE;
  g_cfg.seed = 1;
  g_cfg.show_help = false;
  g_cfg.precisions.clear();
  g_cfg.out_precisions.clear();

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      g_cfg.show_help = true;
      return true;
    }
    if (std::strcmp(argv[i], "--test") == 0 && i + 1 < argc) {
      g_cfg.test_id = std::atoi(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      g_cfg.seed = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 0));
      continue;
    }
    if (std::strcmp(argv[i], "--rm") == 0 && i + 1 < argc) {
      g_cfg.rm = parse_rounding(argv[++i]);
      continue;
    }
    if (std::strcmp(argv[i], "--prec") == 0 && i + 1 < argc) {
      ++i;
      continue;
    }
    std::fprintf(stderr, "Unknown argument '%s'\n", argv[i]);
    return false;
  }

  return true;
}

bool run_single_case(const TestCaseDef& tc, uint32_t base_seed) {
  rng_state = base_seed ^ (0x9e3779b9u * static_cast<uint32_t>(tc.id));

  QuantMatrix a;
  QuantMatrix b;
  QuantMatrix c;
  init_case_inputs(tc, &a, &b, &c);

  std::vector<double> golden = build_golden(tc, a, b, c);
  uint32_t cycles = 0;
  std::vector<double> actual = run_macro_case(tc, a, b, c, &cycles);
  ErrorStats stats = compare_outputs(actual, golden, tc.threshold);

  std::printf("[OTC][case %d] %s\n", tc.id, tc.name);
  std::printf("  shape=%ux%ux%u  A=%s  B=%s  C=%s  D=%s  sparse=%s  rm=%s\n",
              tc.m, tc.n, tc.k,
              prec_name(tc.a_prec),
              prec_name(tc.b_prec),
              prec_name(tc.c_prec),
              prec_name(tc.d_prec),
              sparse_name(tc.sparse_mode),
              rm_name(g_cfg.rm));
  std::printf("  cycles=%u  max_abs=%.6f  mean_abs=%.6f  rms=%.6f  threshold=%.6f\n",
              cycles, stats.max_abs, stats.mean_abs, stats.rms, tc.threshold);

  if (stats.mismatches == 0) {
    std::printf("  PASS\n");
    return true;
  }

  std::printf("  FAIL mismatches=%u\n", stats.mismatches);
  for (uint32_t row = 0, shown = 0; row < tc.m && shown < 8; ++row) {
    for (uint32_t col = 0; col < tc.n && shown < 8; ++col) {
      size_t idx = row * tc.n + col;
      double diff = std::fabs(actual[idx] - golden[idx]);
      if (diff > tc.threshold) {
        std::printf("    mismatch[%u,%u]: actual=% .6f golden=% .6f diff=% .6f\n",
                    row, col, actual[idx], golden[idx], diff);
        ++shown;
      }
    }
  }
  return false;
}

} // namespace

int main(int argc, char* argv[]) {
  if (!parse_args(argc, argv)) {
    print_usage(argv[0]);
    return 1;
  }

  if (g_cfg.show_help) {
    print_usage(argv[0]);
    return 0;
  }

  if (g_cfg.test_id == 0) {
    bool all_passed = true;
    for (const auto& test : kTests) {
      all_passed &= run_single_case(test, g_cfg.seed);
    }
    return all_passed ? 0 : 1;
  }

  const TestCaseDef* test = find_test(g_cfg.test_id);
  if (test == nullptr) {
    std::fprintf(stderr, "Unknown test id %d\n", g_cfg.test_id);
    print_usage(argv[0]);
    return 1;
  }

  return run_single_case(*test, g_cfg.seed) ? 0 : 1;
}
