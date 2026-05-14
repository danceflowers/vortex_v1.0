#pragma once

#include <array>
#include <cstdint>
#include "config_register.h"
#include "fp_types.h"

namespace add4_fp22_rtl {

static constexpr int kInExpWidth = 8;
static constexpr int kInManWidth = 13;
static constexpr int kOutExpWidth = 8;
static constexpr int kOutManWidth = 13;
static constexpr int kLatency = 4;
static constexpr int kSumGrowth = 2;
static constexpr int kGrsWidth = 3;
static constexpr int kAccManWidth = 1 + kSumGrowth + 1 + kOutManWidth + kGrsWidth;
static constexpr int kInFpLen = kInExpWidth + kInManWidth + 1;
static constexpr int kOutFpLen = kOutExpWidth + kOutManWidth + 1;
static constexpr int kInExpBias = (1 << (kInExpWidth - 1)) - 1;
static constexpr int kOutExpBias = (1 << (kOutExpWidth - 1)) - 1;
static constexpr int kBiasDiff = kOutExpBias - kInExpBias;

struct unpacked_t {
  bool sign = false;
  uint32_t exp = 0;
  uint32_t man = 0;
  bool is_zero = false;
  bool is_inf = false;
  bool is_nan = false;
};

// Split an FP22 value into sign/exponent/mantissa plus special-case flags.
inline unpacked_t unpack(uint32_t fp) {
  unpacked_t out{};
  out.sign = (fp >> (kInFpLen - 1)) & 1;
  out.exp = (fp >> kInManWidth) & ((1u << kInExpWidth) - 1);
  auto frac = fp & ((1u << kInManWidth) - 1);
  out.is_zero = (out.exp == 0);
  out.is_inf = (out.exp == ((1u << kInExpWidth) - 1)) && (frac == 0);
  out.is_nan = (out.exp == ((1u << kInExpWidth) - 1)) && (frac != 0);
  out.man = out.is_zero ? frac : ((1u << kInManWidth) | frac);
  return out;
}

inline uint32_t mask_width(int width) {
  if (width >= 32) {
    return 0xffffffffu;
  }
  return (1u << width) - 1u;
}

// Right shift with sticky-bit reporting for IEEE-style alignment.
inline uint32_t shift_right_jam(uint32_t in, uint32_t shamt, bool* sticky) {
  if (sticky) {
    *sticky = false;
  }
  if (shamt == 0) {
    return in;
  }
  if (shamt >= (uint32_t)kAccManWidth) {
    if (sticky) {
      *sticky = in != 0;
    }
    return 0;
  }
  uint32_t lost_mask = mask_width(shamt);
  bool lost = (in & lost_mask) != 0;
  if (sticky) {
    *sticky = lost;
  }
  return in >> shamt;
}

// Align one signed operand to the maximum exponent used by the 4-input add.
inline int32_t align_operand(const unpacked_t& in, uint32_t exp_max) {
  uint32_t exp_diff = exp_max - in.exp;
  uint32_t man_ext = (in.man << (kOutManWidth - kInManWidth + kGrsWidth)) & mask_width(kAccManWidth);
  bool sticky = false;
  uint32_t shifted = shift_right_jam(man_ext, exp_diff, &sticky);
  uint32_t jam = ((shifted >> 1) << 1) | ((shifted & 1u) | (sticky ? 1u : 0u));
  int32_t signed_jam = static_cast<int32_t>(jam);
  return in.sign ? -signed_jam : signed_jam;
}

inline uint32_t mask_acc(uint32_t value) {
  return value & mask_width(kAccManWidth);
}

inline int32_t sign_extend_acc(uint32_t value) {
  value = mask_acc(value);
  if (value & (1u << (kAccManWidth - 1))) {
    return static_cast<int32_t>(value | ~mask_width(kAccManWidth));
  }
  return static_cast<int32_t>(value);
}

inline int leading_zeros_u32(uint32_t value, int width) {
  if (value == 0) {
    return width;
  }
  int count = 0;
  for (int i = width - 1; i >= 0; --i) {
    if (value & (1u << i)) {
      break;
    }
    ++count;
  }
  return count;
}

// Combinational reference model for the 4-input FP22 adder pipeline.
inline uint32_t add4_reference(uint32_t in0,
                               uint32_t in1,
                               uint32_t in2,
                               uint32_t in3,
                               RoundingMode rm) {
  std::array<unpacked_t, 4> unpacked = {unpack(in0), unpack(in1), unpack(in2), unpack(in3)};
  bool invalid = false;
  uint32_t exp_max_01 = unpacked[0].exp > unpacked[1].exp ? unpacked[0].exp : unpacked[1].exp;
  uint32_t exp_max_23 = unpacked[2].exp > unpacked[3].exp ? unpacked[2].exp : unpacked[3].exp;
  uint32_t exp_max = exp_max_01 > exp_max_23 ? exp_max_01 : exp_max_23;
  std::array<int32_t, 4> aligned = {};
  for (int i = 0; i < 4; ++i) {
    invalid |= unpacked[i].is_nan;
    aligned[i] = align_operand(unpacked[i], exp_max);
  }
  if (invalid) {
    return (0xffu << kOutManWidth) | (1u << (kOutManWidth - 1));
  }

  uint32_t a0 = mask_acc(static_cast<uint32_t>(aligned[0]));
  uint32_t a1 = mask_acc(static_cast<uint32_t>(aligned[1]));
  uint32_t a2 = mask_acc(static_cast<uint32_t>(aligned[2]));
  uint32_t a3 = mask_acc(static_cast<uint32_t>(aligned[3]));
  uint32_t csa1_sum_u = mask_acc(a0 ^ a1 ^ a2);
  uint32_t csa1_carry_u = mask_acc((a0 & a1) | (a0 & a2) | (a1 & a2));
  uint32_t csa2_sum_u = mask_acc(csa1_sum_u ^ mask_acc(csa1_carry_u << 1) ^ a3);
  uint32_t csa2_carry_u = mask_acc((csa1_sum_u & mask_acc(csa1_carry_u << 1))
                                 | (csa1_sum_u & a3)
                                 | (mask_acc(csa1_carry_u << 1) & a3));
  int32_t final_sum_ext = sign_extend_acc(mask_acc(csa2_sum_u + mask_acc(csa2_carry_u << 1)));
  bool final_sign = (final_sum_ext < 0);
  uint32_t final_abs = final_sign ? static_cast<uint32_t>(-final_sum_ext) : static_cast<uint32_t>(final_sum_ext);
  bool zero_flag = (final_abs == 0);
  if (zero_flag) {
    return 0;
  }

  uint32_t lzc = static_cast<uint32_t>(leading_zeros_u32(final_abs & mask_width(kAccManWidth - 1), kAccManWidth - 1));
  uint32_t norm_man = ((final_abs & mask_width(kAccManWidth - 1)) << lzc) & mask_width(kAccManWidth - 1);
  int32_t norm_exp = static_cast<int32_t>(exp_max) - static_cast<int32_t>(lzc) + kSumGrowth;
  int32_t exp_bias_conv = norm_exp + kBiasDiff;

  bool rounding = (norm_man >> (kGrsWidth + 2 - 1)) & 1u;
  bool sticky = ((norm_man >> 2) & ((1u << kGrsWidth) - 1u)) != 0;
  uint32_t round_in = (norm_man >> (kAccManWidth - 2 - kOutManWidth)) & ((1u << (kOutManWidth + 1)) - 1u);
  RoundResult rr = do_rounding(round_in, kOutManWidth + 1, final_sign, rounding, sticky, rm);
  uint32_t exp_final = static_cast<uint32_t>(exp_bias_conv + (rr.cout ? 1 : 0));
  bool overflow = (exp_final == ((1u << kOutExpWidth) - 1u));
  if (overflow) {
    return (static_cast<uint32_t>(final_sign) << (kOutFpLen - 1))
         | (((1u << kOutExpWidth) - 1u) << kOutManWidth);
  }
  return (static_cast<uint32_t>(final_sign) << (kOutFpLen - 1))
       | ((exp_final & ((1u << kOutExpWidth) - 1u)) << kOutManWidth)
       | (rr.out & ((1u << kOutManWidth) - 1u));
}

} // namespace add4_fp22_rtl

// Four-stage valid/ready pipeline that sums four FP22 operands per lane.
struct add4_pipe_fp22 {
  struct {
    std::array<uint32_t, 4> in = {};
    bool input_valid = false;
    TensorCoreMeta meta = {};
  } add_input;

  struct {
    std::array<uint32_t, 4> in = {};
    uint32_t passthrough = 0;
    bool valid = false;
    TensorCoreMeta meta = {};
  } r1;

  struct {
    std::array<int32_t, 4> aligned = {};
    uint32_t exp_max = 0;
    bool invalid = false;
    uint32_t passthrough = 0;
    bool valid = false;
    TensorCoreMeta meta = {};
  } r2;

  struct {
    uint32_t abs_value = 0;
    bool sign = false;
    uint32_t exp = 0;
    uint32_t lzc = 0;
    bool invalid = false;
    bool zero_flag = false;
    uint32_t passthrough = 0;
    bool valid = false;
    TensorCoreMeta meta = {};
  } r3;

  struct {
    uint32_t result = 0;
    uint32_t passthrough = 0;
    bool valid = false;
    TensorCoreMeta meta = {};
  } r4;

  void reset() {
    add_input = {};
    r1 = {};
    r2 = {};
    r3 = {};
    r4 = {};
  }

  bool out_valid() const {
    return r4.valid;
  }

  bool active() const {
    return add_input.input_valid || r1.valid || r2.valid || r3.valid || r4.valid;
  }

  bool in_ready(bool out_ready) const {
    const bool s4_ready = out_ready || !r4.valid;
    const bool s3_ready = s4_ready || !r3.valid;
    const bool s2_ready = s3_ready || !r2.valid;
    const bool s1_ready = s2_ready || !r1.valid;
    return s1_ready;
  }

  const uint32_t& out_data() const {
    return r4.result;
  }

  uint32_t out_passthrough() const {
    return r4.passthrough;
  }

  const TensorCoreMeta& out_meta() const {
    return r4.meta;
  }

  // Advance all four stages while preserving backpressure from the consumer.
  void tick(bool out_ready, const Config& g_cfg, uint32_t passthrough) {
    using namespace add4_fp22_rtl;

    const bool s4_ready = out_ready || !r4.valid;
    const bool s3_ready = s4_ready || !r3.valid;
    const bool s2_ready = s3_ready || !r2.valid;
    const bool s1_ready = s2_ready || !r1.valid;

    if (s4_ready) {
      if (r3.valid) {
        if (r3.invalid) {
          r4.result = (0xffu << kOutManWidth) | (1u << (kOutManWidth - 1));
        } else if (r3.zero_flag) {
          r4.result = 0;
        } else {
          uint32_t norm_man = (r3.abs_value << r3.lzc) & mask_width(kAccManWidth - 1);
          int32_t norm_exp = static_cast<int32_t>(r3.exp) - static_cast<int32_t>(r3.lzc) + kSumGrowth;
          int32_t exp_bias_conv = norm_exp + kBiasDiff;

          bool rounding = (norm_man >> (kGrsWidth + 2 - 1)) & 1u;
          bool sticky = ((norm_man >> 2) & ((1u << (kGrsWidth)) - 1u)) != 0;
          uint32_t round_in = (norm_man >> (kAccManWidth - 2 - kOutManWidth)) & ((1u << (kOutManWidth + 1)) - 1u);
          RoundResult rr = do_rounding(round_in, kOutManWidth + 1, r3.sign, rounding, sticky, g_cfg.rm);

          uint32_t exp_final = r3.zero_flag ? 0u : static_cast<uint32_t>(exp_bias_conv + (rr.cout ? 1 : 0));
          bool overflow = (exp_final == ((1u << kOutExpWidth) - 1u));
          if (overflow) {
            r4.result = (static_cast<uint32_t>(r3.sign) << (kOutFpLen - 1))
                      | (((1u << kOutExpWidth) - 1u) << kOutManWidth);
          } else {
            r4.result = (static_cast<uint32_t>(r3.sign) << (kOutFpLen - 1))
                      | ((exp_final & ((1u << kOutExpWidth) - 1u)) << kOutManWidth)
                      | (rr.out & ((1u << kOutManWidth) - 1u));
          }
        }
        r4.passthrough = r3.passthrough;
        r4.valid = true;
        r4.meta = r3.meta;
      } else {
        r4 = {};
      }
    }

    if (s3_ready) {
      if (r2.valid) {
        uint32_t a0 = mask_acc(static_cast<uint32_t>(r2.aligned[0]));
        uint32_t a1 = mask_acc(static_cast<uint32_t>(r2.aligned[1]));
        uint32_t a2 = mask_acc(static_cast<uint32_t>(r2.aligned[2]));
        uint32_t a3 = mask_acc(static_cast<uint32_t>(r2.aligned[3]));

        uint32_t csa1_sum_u = mask_acc(a0 ^ a1 ^ a2);
        uint32_t csa1_carry_u = mask_acc((a0 & a1) | (a0 & a2) | (a1 & a2));
        uint32_t csa2_sum_u = mask_acc(csa1_sum_u ^ mask_acc(csa1_carry_u << 1) ^ a3);
        uint32_t csa2_carry_u = mask_acc((csa1_sum_u & mask_acc(csa1_carry_u << 1))
                                       | (csa1_sum_u & a3)
                                       | (mask_acc(csa1_carry_u << 1) & a3));
        int32_t final_sum_ext = sign_extend_acc(mask_acc(csa2_sum_u + mask_acc(csa2_carry_u << 1)));
        bool final_sign = (final_sum_ext < 0);
        uint32_t final_abs = final_sign ? static_cast<uint32_t>(-final_sum_ext) : static_cast<uint32_t>(final_sum_ext);
        bool zero_flag = (final_abs == 0);

        r3.abs_value = final_abs & mask_width(kAccManWidth - 1);
        r3.sign = final_sign;
        r3.exp = r2.exp_max;
        r3.lzc = zero_flag ? 0u : static_cast<uint32_t>(leading_zeros_u32(r3.abs_value, kAccManWidth - 1));
        r3.invalid = r2.invalid;
        r3.zero_flag = zero_flag;
        r3.passthrough = r2.passthrough;
        r3.valid = true;
        r3.meta = r2.meta;
      } else {
        r3 = {};
      }
    }

    if (s2_ready) {
      if (r1.valid) {
        std::array<unpacked_t, 4> unpacked{};
        uint32_t exp_max_01 = 0;
        uint32_t exp_max_23 = 0;
        uint32_t exp_max = 0;
        bool invalid = false;
        for (int i = 0; i < 4; ++i) {
          unpacked[i] = unpack(r1.in[i]);
          invalid |= unpacked[i].is_nan;
        }
        exp_max_01 = unpacked[0].exp > unpacked[1].exp ? unpacked[0].exp : unpacked[1].exp;
        exp_max_23 = unpacked[2].exp > unpacked[3].exp ? unpacked[2].exp : unpacked[3].exp;
        exp_max = exp_max_01 > exp_max_23 ? exp_max_01 : exp_max_23;
        for (int i = 0; i < 4; ++i) {
          r2.aligned[i] = align_operand(unpacked[i], exp_max);
        }
        r2.exp_max = exp_max;
        r2.invalid = invalid;
        r2.passthrough = r1.passthrough;
        r2.valid = true;
        r2.meta = r1.meta;
      } else {
        r2 = {};
      }
    }

    if (s1_ready) {
      if (add_input.input_valid) {
        r1.in = add_input.in;
        r1.passthrough = passthrough;
        r1.valid = true;
        r1.meta = add_input.meta;
      } else {
        r1 = {};
      }
    }
  }
};
