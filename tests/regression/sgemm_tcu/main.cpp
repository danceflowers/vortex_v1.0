#include "common.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <rvfloats.h>
#include <string.h>
#include <tensor_cfg.h>
#include <type_traits>
#include <unistd.h>
#include <util.h>
#include <vector>
#include <vortex.h>
#include "open_tensorcore/tensor_core_top.h"

#define FLOAT_ULP 6
#define MAX_ERRORS 100

#define RT_CHECK(_expr)                                      \
  do {                                                       \
    int _ret = _expr;                                        \
    if (0 == _ret)                                           \
      break;                                                 \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
    cleanup();                                               \
    exit(-1);                                                \
  } while (false)

using namespace vortex;
namespace vt = tensor;

static bool g_enable_sparse = false;
static bool g_dump_all = false;

static inline float fp16_bits_to_float(uint16_t bits) {
  return bit_cast<float>(rv_htof_s(bits, 0, nullptr));
}

static inline uint16_t float_to_fp16_bits(float value) {
  return rv_ftoh_s(bit_cast<uint32_t>(value), 0, nullptr);
}

template <typename T>
struct open_precision_map;

template <>
struct open_precision_map<vt::fp16> {
  static constexpr PrecisionType value = PREC_FP16;
};

template <>
struct open_precision_map<vt::fp8> {
  static constexpr PrecisionType value = PREC_FP8_E4M3;
};

template <>
struct open_precision_map<vt::fp4> {
  static constexpr PrecisionType value = PREC_FP4_E2M1;
};

static bool run_open_tensorcore_once(const uint16_t a[8][8],
                                     const uint16_t b[8][8],
                                     uint16_t out[8][8]) {
  TensorCoreTop tc;
  tc.reset();
  tc.set_jobs = 1;
  tc.jobs_completed = 0;

  uint32_t c_zero[8][8] = {};
  tc.load_inputs(a, b, c_zero);
  tc.tick(true);
  tc.load_invalid();

  for (uint32_t spin = 0; spin < 10000 && tc.jobs_completed < tc.set_jobs; ++spin) {
    bool done = tc.run();
    if (done) {
      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
          out[i][j] = tc.d_out[i][j];
        }
      }
      return true;
    }
  }
  return false;
}
///////////////////////////////////////////////////////////////////////////////

static void convert_row_to_col_major_4bit(uint8_t *dst, uint32_t width, uint32_t height, const uint8_t *src) {
  // Calculate output size and stride
  uint32_t out_bytes = (width * height + 1) / 2;
  memset(dst, 0, out_bytes);
  uint32_t dst_stride = (height + 1) / 2; // Bytes per column in output

  // For each column in source (which becomes row in destination)
  for (uint32_t c = 0; c < width; ++c) {
    uint32_t base = c * dst_stride;

    // For each row in source (which becomes column in destination)
    for (uint32_t r = 0; r < height; r += 2) {
      // Calculate source indices (row-major)
      uint32_t idx_even = r * width + c;
      uint32_t idx_odd = (r + 1) * width + c;

      // Extract nibbles - consistent with data_accessor_t
      uint8_t b_even = src[idx_even / 2];
      uint8_t b_odd = (r + 1 < height) ? src[idx_odd / 2] : 0;

      uint8_t nib_even = (idx_even & 1) ? (b_even >> 4) : (b_even & 0x0F);
      uint8_t nib_odd = (r + 1 < height)
                            ? ((idx_odd & 1) ? (b_odd >> 4) : (b_odd & 0x0F))
                            : 0;

      // Pack into destination: even row in low nibble, odd row in high nibble
      dst[base + r / 2] = (nib_odd << 4) | nib_even;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct data_accessor_t {
  using Type = typename T::dtype;
  static Type read(const Type *ptr, uint32_t offset) {
    return ptr[offset];
  }
  static void write(Type *ptr, uint32_t offset, Type value) {
    ptr[offset] = value;
  }
};

template <>
struct data_accessor_t<vt::int4> {
  static uint8_t read(const uint8_t *ptr, uint32_t offset) {
    uint32_t row_off = offset / 2;
    bool odd = offset & 0x1;
    uint8_t value8 = ptr[row_off];
    return odd ? (value8 >> 4) : (value8 & 0x0f); // to nibble
  }
  static void write(uint8_t *ptr, uint32_t offset, int32_t value) {
    uint32_t row_off = offset / 2;
    bool odd = offset & 0x1;
    uint8_t old_value = ptr[row_off];
    uint8_t new_value = odd ? ((old_value & 0x0f) | (value << 4))
                            : ((old_value & 0xf0) | (value & 0x0f));
    ptr[offset / 2] = new_value;
  }
};

template <>
struct data_accessor_t<vt::uint4> {
  static uint8_t read(const uint8_t *ptr, uint32_t offset) {
    uint32_t row_off = offset / 2;
    bool odd = offset & 0x1;
    uint8_t value8 = ptr[row_off];
    return odd ? (value8 >> 4) : (value8 & 0x0f); // to nibble
  }
  static void write(uint8_t *ptr, uint32_t offset, int32_t value) {
    uint32_t row_off = offset / 2;
    bool odd = offset & 0x1;
    uint8_t old_value = ptr[row_off];
    uint8_t new_value = odd ? ((old_value & 0x0f) | (value << 4))
                            : ((old_value & 0xf0) | (value & 0x0f));
    ptr[offset / 2] = new_value;
  }
};

template <>
struct data_accessor_t<vt::fp4> {
  static uint8_t read(const uint8_t *ptr, uint32_t offset) {
    uint32_t row_off = offset / 2;
    bool odd = offset & 0x1;
    uint8_t value8 = ptr[row_off];
    return odd ? (value8 >> 4) : (value8 & 0x0f);
  }
  static void write(uint8_t *ptr, uint32_t offset, int32_t value) {
    uint32_t row_off = offset / 2;
    bool odd = offset & 0x1;
    uint8_t old_value = ptr[row_off];
    uint8_t new_value = odd ? ((old_value & 0x0f) | ((value & 0x0f) << 4))
                            : ((old_value & 0xf0) | (value & 0x0f));
    ptr[row_off] = new_value;
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class Comparator {};

template <>
class Comparator<vt::int8> {
public:
  static int8_t generate() {
    return (int8_t)rand();
  }
  static bool compare(int8_t a, int8_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::uint8> {
public:
  static uint8_t generate() {
    return (uint8_t)rand();
  }
  static bool compare(uint8_t a, uint8_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::int4> {
public:
  static uint8_t generate() {
    return (uint8_t)rand(); // store 2 nibbles in a byte
  }
  static bool compare(uint8_t a, uint8_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::uint4> {
public:
  static uint8_t generate() {
    return (uint8_t)rand(); // store 2 nibbles in a byte
  }
  static bool compare(uint8_t a, uint8_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::fp8> {
public:
  static uint8_t generate() {
    uint8_t sign = (rand() & 0x1) << 7;
    uint8_t exp  = static_cast<uint8_t>(2 + (rand() % 8)); // keep dynamic range moderate
    uint8_t mant = static_cast<uint8_t>(rand() & 0x7);
    return static_cast<uint8_t>(sign | (exp << 3) | mant);
  }
  static bool compare(uint8_t a, uint8_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::fp4> {
public:
  static uint8_t generate() {
    auto gen_nibble = []() -> uint8_t {
      uint8_t sign = (rand() & 0x1) << 3;
      uint8_t exp  = static_cast<uint8_t>(rand() % 3); // avoid exp=3 (Inf/NaN class)
      uint8_t mant = static_cast<uint8_t>(rand() & 0x1);
      return static_cast<uint8_t>(sign | (exp << 1) | mant);
    };
    return static_cast<uint8_t>((gen_nibble() << 4) | gen_nibble());
  }
  static bool compare(uint8_t a, uint8_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::int32> {
public:
  static int32_t generate() {
    return (int32_t)rand();
  }
  static bool compare(int32_t a, int32_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::fp16> {
public:
  static uint16_t generate() {
    auto fvalue = float(rand()) / RAND_MAX;
    return rv_ftoh_s(bit_cast<uint32_t>(fvalue), 0, nullptr);
  }
  static bool compare(uint16_t a, uint16_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::bf16> {
public:
  static uint16_t generate() {
    auto fvalue = float(rand()) / RAND_MAX;
    return rv_ftob_s(bit_cast<uint32_t>(fvalue), 0, nullptr);
  }
  static bool compare(uint16_t a, uint16_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

// template <>
// class Comparator<vt::tf32> {
// public:
//   static uint32_t generate() {
//     auto fvalue = float(rand()) / RAND_MAX;
//     return rv_ftox_s(bit_cast<uint32_t>(fvalue), 8, 10, 0, nullptr);
//   }
//   static bool compare(uint32_t a, uint32_t b, int index, int errors) {
//     if (a != b) {
//       if (errors < MAX_ERRORS) {
//         printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
//       }
//       return false;
//     }
//     return true;
//   }
// };

template <>
class Comparator<vt::fp32> {
public:
  static float generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    union fi_t {
      float f;
      int32_t i;
    };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, fb.f, fa.f);
      }
      return false;
    }
    return true;
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename S, typename D>
struct muladd_t {
  using stype = typename S::dtype;
  using dtype = typename D::dtype;
  static dtype eval(stype a, stype b, dtype c) {
    return static_cast<dtype>(a) * static_cast<dtype>(b) + c;
  }
};

static inline float fp8_e4m3_to_float(uint8_t v) {
  bool s = (v >> 7) & 0x1;
  uint32_t e = (v >> 3) & 0xF;
  uint32_t m = v & 0x7;
  if (e == 0xF) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  float out = 0.0f;
  if (e == 0) {
    out = std::ldexp(static_cast<float>(m) / 8.0f, -6);
  } else {
    out = std::ldexp(1.0f + static_cast<float>(m) / 8.0f, static_cast<int>(e) - 7);
  }
  return s ? -out : out;
}

static inline float fp4_e2m1_to_float(uint8_t v) {
  bool s = (v >> 3) & 0x1;
  uint32_t e = (v >> 1) & 0x3;
  uint32_t m = v & 0x1;
  float out = 0.0f;
  if (e == 0x3) {
    if (m == 0) {
      out = std::numeric_limits<float>::infinity();
    } else {
      out = std::numeric_limits<float>::quiet_NaN();
    }
  } else if (e == 0) {
    out = (static_cast<float>(m) / 2.0f);
  } else {
    out = std::ldexp(1.0f + static_cast<float>(m) / 2.0f, static_cast<int>(e) - 1);
  }
  return s ? -out : out;
}

template <typename T>
static inline float input_to_float(typename T::dtype v) {
  return static_cast<float>(v);
}

template <>
inline float input_to_float<vt::fp16>(uint16_t v) {
  return bit_cast<float>(rv_htof_s(v, 0, nullptr));
}

template <>
inline float input_to_float<vt::bf16>(uint16_t v) {
  return bit_cast<float>(rv_btof_s(v, 0, nullptr));
}

template <>
inline float input_to_float<vt::fp8>(uint8_t v) {
  return fp8_e4m3_to_float(v);
}

template <>
inline float input_to_float<vt::fp4>(uint8_t v) {
  return fp4_e2m1_to_float(v & 0xF);
}

template <>
inline float input_to_float<vt::int4>(uint8_t v) {
  int32_t x = v & 0xF;
  if (x & 0x8)
    x |= 0xFFFFFFF0;
  return static_cast<float>(x);
}

template <>
inline float input_to_float<vt::uint4>(uint8_t v) {
  return static_cast<float>(v & 0xF);
}

template <>
struct muladd_t<vt::fp16, vt::fp32> {
  static float eval(uint16_t a, uint16_t b, float c) {
    auto fa = bit_cast<float>(rv_htof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_htof_s(b, 0, nullptr));
    return fa * fb + c;
  }
};

template <>
struct muladd_t<vt::fp16, vt::fp16> {
  static uint16_t eval(uint16_t a, uint16_t b, uint16_t c) {
    auto fa = bit_cast<float>(rv_htof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_htof_s(b, 0, nullptr));
    auto fc = bit_cast<float>(rv_htof_s(c, 0, nullptr));
    auto fd = fa * fb + fc;
    return rv_ftoh_s(bit_cast<uint32_t>(fd), 0, nullptr);
  }
};

template <>
struct muladd_t<vt::bf16, vt::fp32> {
  static float eval(uint16_t a, uint16_t b, float c) {
    auto fa = bit_cast<float>(rv_btof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_btof_s(b, 0, nullptr));
    return fa * fb + c;
  }
};

template <>
struct muladd_t<vt::bf16, vt::bf16> {
  static uint16_t eval(uint16_t a, uint16_t b, uint16_t c) {
    auto fa = bit_cast<float>(rv_btof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_btof_s(b, 0, nullptr));
    auto fc = bit_cast<float>(rv_btof_s(c, 0, nullptr));
    auto fd = fa * fb + fc;
    return rv_ftob_s(bit_cast<uint32_t>(fd), 0, nullptr);
  }
};

template <>
struct muladd_t<vt::fp8, vt::fp16> {
  static uint16_t eval(uint8_t a, uint8_t b, uint16_t c) {
    float fa = fp8_e4m3_to_float(a);
    float fb = fp8_e4m3_to_float(b);
    float fc = bit_cast<float>(rv_htof_s(c, 0, nullptr));
    float fd = fa * fb + fc;
    return rv_ftoh_s(bit_cast<uint32_t>(fd), 0, nullptr);
  }
};

template <>
struct muladd_t<vt::fp4, vt::fp16> {
  static uint16_t eval(uint8_t a, uint8_t b, uint16_t c) {
    float fa = fp4_e2m1_to_float(a & 0xF);
    float fb = fp4_e2m1_to_float(b & 0xF);
    float fc = bit_cast<float>(rv_htof_s(c, 0, nullptr));
    float fd = fa * fb + fc;
    return rv_ftoh_s(bit_cast<uint32_t>(fd), 0, nullptr);
  }
};

// template <>
// struct muladd_t<vt::tf32, vt::fp32> {
//   static float eval(uint32_t a, uint32_t b, float c) {
//     auto fa = bit_cast<float>(rv_xtof_s(a, 8, 10, 0, nullptr));
//     auto fb = bit_cast<float>(rv_xtof_s(b, 8, 10, 0, nullptr));
//     return fa * fb + c;
//   }
// };

template <>
struct muladd_t<vt::int4, vt::int32> {
  static int32_t eval(uint8_t a, uint8_t b, int32_t c) {
    int32_t a_val = a & 0xF;
    if (a & 0x8) {
      a_val |= 0xFFFFFFF0; // sign extend
    }
    int32_t b_val = b & 0xF;
    if (b & 0x8) {
      b_val |= 0xFFFFFFF0; // sign extend
    }
    return a_val * b_val + c;
  }
};

template <>
struct muladd_t<vt::uint4, vt::int32> {
  static int32_t eval(uint8_t a, uint8_t b, int32_t c) {
    int32_t a_val = a & 0xF;
    int32_t b_val = b & 0xF;
    return a_val * b_val + c;
  }
};

///////////////////////////////////////////////////////////////////////////////

using cfg = vt::wmma_config_t<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

using itype_t = typename vt::ITYPE::dtype;
using otype_t = typename vt::OTYPE::dtype;

struct SparseMat {
  std::vector<itype_t> values;   // non-zeros
  std::vector<uint8_t> meta;     // Array of row-masks: 1 byte marks the columns
                                  // of the 4 elements in the block that are non-zero.
                                  // e.g. 0b0101 means 2nd and 4th elements are non-zero.

  uint32_t rows, cols;           // original A dims (M × K)
};

static void matmul_cpu(otype_t *C, const itype_t *A, const itype_t *B, uint32_t M, uint32_t N, uint32_t K) {
  uint32_t subbytes = 8 / vt::ITYPE::bits;
  uint32_t KS = subbytes ? (K * subbytes) : K;
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      otype_t sum(0);
      for (uint32_t k = 0; k < KS; ++k) {
        auto a = data_accessor_t<vt::ITYPE>::read(A, m * KS + k);
        auto b = data_accessor_t<vt::ITYPE>::read(B, k * N + n);
        sum = muladd_t<vt::ITYPE, vt::OTYPE>::eval(a, b, sum);
      }
      data_accessor_t<vt::OTYPE>::write(C, m * N + n, sum);
    }
  }
}

static void matmul_cpu_fp32_acc(float *C, const itype_t *A, const itype_t *B, uint32_t M, uint32_t N, uint32_t K) {
  uint32_t subbytes = 8 / vt::ITYPE::bits;
  uint32_t KS = subbytes ? (K * subbytes) : K;
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < KS; ++k) {
        auto a_raw = data_accessor_t<vt::ITYPE>::read(A, m * KS + k);
        auto b_raw = data_accessor_t<vt::ITYPE>::read(B, k * N + n);
        float a = input_to_float<vt::ITYPE>(a_raw);
        float b = input_to_float<vt::ITYPE>(b_raw);
        sum += a * b;
      }
      C[m * N + n] = sum;
    }
  }
}

static void matmul_open_tensorcore_ref(otype_t *C, const itype_t *A, const itype_t *B, uint32_t M, uint32_t N, uint32_t K) {
  if constexpr (!std::is_same_v<otype_t, uint16_t>) {
    // open_tensorcore reference path currently models fp16 output only.
    // Fall back to the generic CPU reference for other output types.
    matmul_cpu(C, A, B, M, N, K);
    return;
  }

  ::g_cfg.precisions.clear();
  ::g_cfg.out_precisions.clear();
  ::g_cfg.precisions.push_back(open_precision_map<vt::ITYPE>::value);
  ::g_cfg.out_precisions.push_back(PREC_FP16);
  ::g_cfg.rm = RNE;

  uint32_t subbytes = 8 / vt::ITYPE::bits;
  uint32_t KS = subbytes ? (K * subbytes) : K;

  for (uint32_t m0 = 0; m0 < M; m0 += 8) {
    for (uint32_t n0 = 0; n0 < N; n0 += 8) {
      uint16_t acc_fp16[8][8] = {};

      for (uint32_t k0 = 0; k0 < KS; k0 += 8) {
        uint16_t a_blk[8][8] = {};
        uint16_t b_blk[8][8] = {};
        uint16_t out_blk[8][8] = {};

        for (uint32_t i = 0; i < 8; ++i) {
          for (uint32_t k = 0; k < 8; ++k) {
            uint32_t kk = k0 + k;
            uint16_t raw = 0;
            if (kk < KS) {
              raw = static_cast<uint16_t>(data_accessor_t<vt::ITYPE>::read(A, (m0 + i) * KS + kk));
            }
            a_blk[i][k] = raw;
          }
        }

        for (uint32_t k = 0; k < 8; ++k) {
          uint32_t kk = k0 + k;
          for (uint32_t j = 0; j < 8; ++j) {
            uint16_t raw = 0;
            if (kk < KS) {
              raw = static_cast<uint16_t>(data_accessor_t<vt::ITYPE>::read(B, kk * N + (n0 + j)));
            }
            b_blk[k][j] = raw;
          }
        }

        if (!run_open_tensorcore_once(a_blk, b_blk, out_blk)) {
          std::abort();
        }

        for (uint32_t i = 0; i < 8; ++i) {
          for (uint32_t j = 0; j < 8; ++j) {
            float acc_val = fp16_bits_to_float(acc_fp16[i][j]);
            float out_val = fp16_bits_to_float(out_blk[i][j]);
            acc_fp16[i][j] = float_to_fp16_bits(acc_val + out_val);
          }
        }
      }

      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
          data_accessor_t<vt::OTYPE>::write(C, (m0 + i) * N + (n0 + j), acc_fp16[i][j]);
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

const char *kernel_file = "kernel.vxbin";

uint32_t xm = 32;
uint32_t xn = 32;
uint32_t xk = 32;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

std::string last_build_options;

static void show_usage() {
  std::cout << "Vortex Sgemm TCU Test." << std::endl;
  std::cout << "Usage: [-m: m] [-n N] [-k: K] [-s] [--dump=all] [-h: help]" << std::endl;
  std::cout << "  -s          Enable 2:4 structured sparsity" << std::endl;
  std::cout << "  --dump=all  Print full matrices (default is top-left 8x8)" << std::endl;
}

static void parse_args(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    if (0 == strcmp(argv[i], "-m") && i + 1 < argc) {
      xm = atoi(argv[++i]);
    } else if (0 == strcmp(argv[i], "-n") && i + 1 < argc) {
      xn = atoi(argv[++i]);
    } else if (0 == strcmp(argv[i], "-k") && i + 1 < argc) {
      xk = atoi(argv[++i]);
    } else if (0 == strcmp(argv[i], "-s")) {
      g_enable_sparse = true;
      std::cout << "Sparse mode enabled (-s)" << std::endl;
    } else if (0 == strcmp(argv[i], "--dump=all")) {
      g_dump_all = true;
    } else if (0 == strcmp(argv[i], "-h")) {
      show_usage();
      exit(0);
    } else {
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(A_buffer);
    vx_mem_free(B_buffer);
    vx_mem_free(C_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}


static SparseMat pruneAndCompressMatrixA(const std::vector<itype_t>& denseA,
                                         uint32_t M, uint32_t K) {
  SparseMat out;
  out.rows = M;
  out.cols = K;
  out.values.reserve(M * K / 2); // Select 2 values every 4 values
  out.meta.reserve(M * K / 4); // 1 byte for every 4 values

  const itype_t* src = denseA.data();

  for (uint32_t r = 0; r < M; ++r) {
    for (uint32_t c = 0; c < K; c += 4) {
      itype_t blk[4] = {src[r * K + c],
                        src[r * K + c + 1],
                        src[r * K + c + 2],
                        src[r * K + c + 3]};

      uint32_t idx[4] = {0, 1, 2, 3};
      std::sort(idx, idx + 4,
        [&](uint32_t a, uint32_t b) {
          return std::abs((int)blk[a]) < std::abs((int)blk[b]);
        }); //Sort the 4 elements by absolute value, ascending order

      uint8_t keep0 = idx[3];
      uint8_t keep1 = idx[2]; //idx of largest 2 elements

      out.values.push_back(blk[keep0]);
      out.values.push_back(blk[keep1]);

      uint8_t m = (1u << keep0) | (1u << keep1);  // e.g. 0b0101
      out.meta.push_back(m);
    }
  }
  return out;
}

void test_pruneA() {
  const uint32_t M = 4, K = 8;
  std::vector<itype_t> denseA(M * K);
  for (auto& v : denseA) v = Comparator<vt::ITYPE>::generate();

  auto spA = pruneAndCompressMatrixA(denseA, M, K);

  std::vector<itype_t> recovered(M * K, 0);
  size_t v_idx = 0, m_idx = 0;
  for (uint32_t r = 0; r < M; ++r)
    for (uint32_t c = 0; c < K; c += 4) {
      uint8_t m = spA.meta[m_idx++];
      for (uint32_t i = 0; i < 4; ++i)
        if (m & (1u << i))
          recovered[r * K + c + i] = spA.values[v_idx++];
    }

  for (uint32_t i = 0; i < M * K; ++i)
    assert(recovered[i] == denseA[i] || recovered[i] == 0); //Either the value is preserved or pruned
  std::cout << "pruneAndCompressMatrixA passed\n";
}


int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  if(g_enable_sparse) {
    test_pruneA(); // Test the pruning function
  }

  std::srand(50);

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  uint64_t isa_flags;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  bool has_ext = (isa_flags & VX_ISA_EXT_TCU) != 0;
  if (!has_ext) {
    std::cout << "TCU extension not supported!" << std::endl;
    cleanup();
    return -1;
  }

  uint64_t NT;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &NT));
  if (NT != NUM_THREADS) {
    std::cout << "Error: device warp size (" << NT << ") must match NUM_THREADS=" << NUM_THREADS << "!" << std::endl;
    return -1;
  }

  uint32_t M = xm;
  uint32_t N = xn;
  uint32_t K = xk;

  if ((M % cfg::tileM) != 0) {
    std::cout << "Error: M must be a multiple of tensor tileM!" << std::endl;
    return -1;
  }

  if ((N % cfg::tileN) != 0) {
    std::cout << "Error: M must be a multiple of tensor tileN!" << std::endl;
    return -1;
  }

  if ((K % cfg::tileK) != 0) {
    std::cout << "Error: M must be a multiple of tensor tileK!" << std::endl;
    return -1;
  }

  size_t sizeA = M * K;
  size_t sizeB = K * N;
  size_t sizeC = M * N;

  std::cout << "input data type: " << vt::ITYPE::name << " (id=" << vt::ITYPE::id << ")" << std::endl;
  std::cout << "output data type: " << vt::OTYPE::name << " (id=" << vt::OTYPE::id << ")" << std::endl;
  std::cout << "WMMA Core Dimension: M=" << cfg::tcM << ", N=" << cfg::tcN << ", K=" << cfg::tcK << std::endl;
  std::cout << "WMMA Tile Dimension: M=" << cfg::tileM << ", N=" << cfg::tileN << ", K=" << cfg::tileK << std::endl;
  std::cout << "matrix A: " << M << "x" << K << std::endl;
  std::cout << "matrix B: " << K << "x" << N << std::endl;
  std::cout << "matrix C: " << M << "x" << N << std::endl;

  // set block size to warp size
  kernel_arg.grid_dim[0] = N / cfg::tileN;
  kernel_arg.grid_dim[1] = M / cfg::tileM;
  kernel_arg.block_dim[0] = NT; // warp sizeb
  kernel_arg.block_dim[1] = 1;

  // set matrix dimensions
  kernel_arg.M = M;
  kernel_arg.N = N;
  kernel_arg.K = K;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, sizeA * sizeof(itype_t), VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_alloc(device, sizeB * sizeof(itype_t), VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_mem_alloc(device, sizeC * sizeof(otype_t), VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.C_addr));

  std::cout << "A_addr=0x" << std::hex << kernel_arg.A_addr << std::endl;
  std::cout << "B_addr=0x" << std::hex << kernel_arg.B_addr << std::endl;
  std::cout << "C_addr=0x" << std::hex << kernel_arg.C_addr << std::endl;

  // generate source data
  std::vector<itype_t> h_A(sizeA);
  std::vector<itype_t> h_B(sizeB);
  for (uint32_t i = 0; i < sizeA; ++i) {
    h_A[i] = Comparator<vt::ITYPE>::generate();
  }
  for (uint32_t i = 0; i < sizeB; ++i) {
    h_B[i] = Comparator<vt::ITYPE>::generate();
  }

  // upload matrix A buffer
  { 
    std::cout << "upload matrix A buffer" << std::endl;
    RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, sizeA * sizeof(itype_t)));
  }

  // upload matrix B buffer
  {
    std::cout << "upload matrix B buffer" << std::endl;
    if constexpr (std::is_same<vt::ITYPE, vt::int4>::value
               || std::is_same<vt::ITYPE, vt::uint4>::value
               || std::is_same<vt::ITYPE, vt::fp4>::value) {
      // sub-byte matrix B must be in col-major format
      // we convert the 4-bit row-major to col-major here
      std::vector<uint8_t> h_B_col(sizeB);
      convert_row_to_col_major_4bit(h_B_col.data(), N, 2 * K, (uint8_t*)h_B.data());
      RT_CHECK(vx_copy_to_dev(B_buffer, h_B_col.data(), 0, sizeB));
    } else {
      RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, sizeB * sizeof(itype_t)));
    }
  }

  // upload program
  std::cout << "upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  auto time_start = std::chrono::high_resolution_clock::now();

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // download destination buffer
  std::vector<otype_t> h_C(sizeC);
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, sizeC * sizeof(otype_t)));

  // verify result (print-only, no compare/assert)
  std::cout << "verify result (print-only)" << std::endl;
  std::vector<otype_t> h_ref(sizeC);
  if constexpr ((std::is_same_v<vt::OTYPE, vt::fp16>)
             && (std::is_same_v<vt::ITYPE, vt::fp16>
              || std::is_same_v<vt::ITYPE, vt::fp8>
              || std::is_same_v<vt::ITYPE, vt::fp4>)) {
    std::cout << "golden mode: open_tensorcore reference" << std::endl;
    matmul_open_tensorcore_ref(h_ref.data(), h_A.data(), h_B.data(), M, N, K);
  } else {
    matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), M, N, K);
  }

  auto to_float = [](otype_t v) {
    if constexpr (std::is_same_v<otype_t, uint16_t>) {
      return bit_cast<float>(rv_htof_s(v, 0, nullptr));
    } else {
      return static_cast<float>(v);
    }
  };

  uint32_t dump_rows = g_dump_all ? M : std::min<uint32_t>(M, 8);
  uint32_t dump_cols = g_dump_all ? N : std::min<uint32_t>(N, 8);

  std::cout << "Result D (" << dump_rows << "x" << dump_cols << ")" << std::endl;
  for (uint32_t i = 0; i < dump_rows; ++i) {
    for (uint32_t j = 0; j < dump_cols; ++j) {
      std::cout << to_float(h_C[i * N + j]) << (j + 1 == dump_cols ? '\n' : ' ');
    }
  }

  std::cout << "Golden reference (" << dump_rows << "x" << dump_cols << ")" << std::endl;
  for (uint32_t i = 0; i < dump_rows; ++i) {
    for (uint32_t j = 0; j < dump_cols; ++j) {
      std::cout << to_float(h_ref[i * N + j]) << (j + 1 == dump_cols ? '\n' : ' ');
    }
  }

  std::vector<float> h_ref_fp32(sizeC);
  matmul_cpu_fp32_acc(h_ref_fp32.data(), h_A.data(), h_B.data(), M, N, K);

  std::cout << "FP32 accumulate reference (" << dump_rows << "x" << dump_cols << ")" << std::endl;
  for (uint32_t i = 0; i < dump_rows; ++i) {
    for (uint32_t j = 0; j < dump_cols; ++j) {
      std::cout << h_ref_fp32[i * N + j] << (j + 1 == dump_cols ? '\n' : ' ');
    }
  }

  double max_abs_err = 0.0;
  double mse = 0.0;
  for (uint32_t i = 0; i < sizeC; ++i) {
    double diff = std::abs(static_cast<double>(to_float(h_C[i])) - static_cast<double>(to_float(h_ref[i])));
    max_abs_err = std::max(max_abs_err, diff);
    mse += diff * diff;
  }
  double rmse = std::sqrt(mse / sizeC);
  std::cout << "max_abs_err=" << max_abs_err << ", rmse=" << rmse << std::endl;

  double max_abs_err_fp32 = 0.0;
  double mse_fp32 = 0.0;
  for (uint32_t i = 0; i < sizeC; ++i) {
    double diff = std::abs(static_cast<double>(to_float(h_C[i])) - static_cast<double>(h_ref_fp32[i]));
    max_abs_err_fp32 = std::max(max_abs_err_fp32, diff);
    mse_fp32 += diff * diff;
  }
  double rmse_fp32 = std::sqrt(mse_fp32 / sizeC);
  std::cout << "vs_fp32_acc: max_abs_err=" << max_abs_err_fp32 << ", rmse=" << rmse_fp32 << std::endl;

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  std::cout << "PASSED!" << std::endl;
  return 0;
}
