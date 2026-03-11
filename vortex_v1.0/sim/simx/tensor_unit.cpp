
// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensor_unit.h"
#include "tensor_cfg.h"
#include <rvfloats.h>
#include "core.h"
#include "open_tensorcore/tensor_core_top.h"

using namespace vortex;

namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<NUM_THREADS>;

inline uint64_t nan_box(uint32_t value) {
  return value | 0xffffffff00000000;
}


uint16_t unpack_word_lane(uint32_t fmt_s, uint32_t a_word, uint32_t lane) {
    if (fmt_s == vt::fp16::id) {
        // fp16: 每次取 16 bit，lane 范围 0~1
        if (lane > 1) return 0; // 防止越界
        return (a_word >> (lane * 16)) & 0xFFFF;
    } 
    else if (fmt_s == vt::fp8::id) {
        // fp8: 每次取 8 bit，lane 范围 0~3
        if (lane > 3) return 0; // 防止越界
        return (a_word >> (lane * 8)) & 0xFF;
    } 
    else if (fmt_s == vt::fp4::id) {
        // fp4: 8 packed nibbles in one 32-bit word
        if (lane > 7) return 0;
        return (a_word >> (lane * 4)) & 0xF;
    }
    
    // 如果传入了未知的 fmt_s，默认返回 0
    return 0;
}


inline bool use_open_tensorcore(uint32_t fmt_s, uint32_t fmt_d) {
  if (fmt_d != vt::fp16::id)
    return false;
  // The open-tensorcore path below is hard-wired for an 8x8 output tile
  // mapped to 32 threads (32 packed fp16 output words). For other warp sizes
  // this mapping does not hold.
  if constexpr (NUM_THREADS != 32) {
    return false;
  }
  return (fmt_s == vt::fp16::id || fmt_s == vt::fp4::id || fmt_s == vt::fp8::id);
}



float fp16_to_float(uint32_t input) {
    uint16_t h = input & 0xFFFF;
    return bit_cast<float>(rv_htof_s(h, 0, nullptr));
}

static inline uint16_t fp16_add(uint16_t a, uint16_t b) {
    auto xa = rv_htof_s(a, 0, nullptr);
    auto xb = rv_htof_s(b, 0, nullptr);
    auto xc = rv_fadd_s(xa, xb, 0, nullptr);
    return rv_ftoh_s(xc, 0, nullptr);
}


uint16_t float_to_fp16(float input) {
    return rv_ftoh_s(bit_cast<uint32_t>(input), 0, nullptr);
}
//=============================================================================

template <typename It, typename Ot>
struct FMA {
  using itype = typename It::dtype;
  using otype = typename Ot::dtype;
  static otype eval(itype a, itype b, otype c) {
    return static_cast<otype>(a) * static_cast<otype>(b) + c;
  }
};

template <>
struct FMA<vt::fp16, vt::fp32> {
  static float eval(uint16_t a, uint16_t b, float c) {
    auto xa = rv_htof_s(a, 0, nullptr);
    auto xb = rv_htof_s(b, 0, nullptr);
    auto xab= rv_fmul_s(xa, xb, 0, nullptr);
    auto xc = bit_cast<uint32_t>(c);
    auto xd = rv_fadd_s(xab, xc, 0, nullptr);
    return bit_cast<float>(xd);
  }
};

template <>
struct FMA<vt::fp16, vt::fp16> {
  static uint16_t eval(uint16_t a, uint16_t b, uint16_t c) {
    auto xa = rv_htof_s(a, 0, nullptr);
    auto xb = rv_htof_s(b, 0, nullptr);
    auto xc = rv_htof_s(c, 0, nullptr);
    auto xd = rv_fmadd_s(xa, xb, xc, 0, nullptr);
    auto xh = rv_ftoh_s(xd, 0, nullptr);
    return xh;
  }
};

template <>
struct FMA<vt::bf16, vt::fp32> {
  static float eval(uint16_t a, uint16_t b, float c) {
    auto xa = rv_btof_s(a, 0, nullptr);
    auto xb = rv_btof_s(b, 0, nullptr);
    auto xab= rv_fmul_s(xa, xb, 0, nullptr);
    auto xc = bit_cast<uint32_t>(c);
    auto xd = rv_fadd_s(xab, xc, 0, nullptr);
    return bit_cast<float>(xd);
  }
};

template <>
struct FMA<vt::bf16, vt::bf16> {
  static uint16_t eval(uint16_t a, uint16_t b, uint16_t c) {
    auto xa = rv_btof_s(a, 0, nullptr);
    auto xb = rv_btof_s(b, 0, nullptr);
    auto xc = rv_btof_s(c, 0, nullptr);
    auto xd = rv_fmadd_s(xa, xb, xc, 0, nullptr);
    auto xh = rv_ftob_s(xd, 0, nullptr);
    return xh;
  }
};

template <typename It, typename Ot>
struct FEDP {
  using itype = typename It::dtype;
  using otype = typename Ot::dtype;
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val) {
  constexpr uint32_t i_ratio = sizeof(uint32_t) / sizeof(itype);
  static_assert(i_ratio * sizeof(itype) == sizeof(uint32_t), "FEDP: tcK * i_ratio must be <= 32");
  auto acc = bit_cast<otype>(c_val);
  for (uint32_t z = 0; z < cfg::tcK; ++z) {
    auto a = reinterpret_cast<const itype *>(&a_row[z].u32);
    auto b = reinterpret_cast<const itype *>(&b_col[z].u32);
    for (uint32_t i = 0; i < i_ratio; ++i) {
      acc = FMA<It, Ot>::eval(a[i], b[i], acc);
    }
  }
  return bit_cast<uint32_t>(acc);
  }
};

template <>
struct FEDP<vt::int4, vt::int32>{
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val) {
    auto acc = bit_cast<int32_t>(c_val);
    for (uint32_t z = 0; z < cfg::tcK; ++z) {
      auto a = a_row[z].u32;
      auto b = b_col[z].u32;
      for (uint32_t i = 0; i < 8; ++i) { // 8 * 4 bits = 32 bits
        int32_t a_val = (a >> (i * 4)) & 0xF;
        int32_t b_val = (b >> (i * 4)) & 0xF;
        if (a_val & 0x8) {
          a_val |= 0xFFFFFFF0;
        }
        if (b_val & 0x8) {
          b_val |= 0xFFFFFFF0;
        }
        acc += a_val * b_val;
      }
    }
    return bit_cast<uint32_t>(acc);
  }
};

template <>
struct FEDP<vt::uint4, vt::int32>{
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val) {
    auto acc = bit_cast<int32_t>(c_val);
    for (uint32_t z = 0; z < cfg::tcK; ++z) {
      auto a = a_row[z].u32;
      auto b = b_col[z].u32;
      for (uint32_t i = 0; i < 8; ++i) { // 8 * 4 bits = 32 bits
        int32_t a_val = (a >> (i * 4)) & 0xF;
        int32_t b_val = (b >> (i * 4)) & 0xF;
        acc += a_val * b_val;
      }
    }
    return bit_cast<uint32_t>(acc);
  }
};

using PFN_FEDP = uint32_t (*)(const reg_data_t*, const reg_data_t*, uint32_t);

static PFN_FEDP select_FEDP(uint32_t IT, uint32_t OT) {
  switch (OT) {
  case vt::fp32::id:
    switch (IT) {
    case vt::fp16::id:
      return FEDP<vt::fp16, vt::fp32>::eval;
    case vt::bf16::id:
      return FEDP<vt::bf16, vt::fp32>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::fp16::id:
    switch (IT) {
    case vt::fp16::id:
      return FEDP<vt::fp16, vt::fp16>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::bf16::id:
    switch (IT) {
    case vt::bf16::id:
      return FEDP<vt::bf16, vt::bf16>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::int32::id:
    switch (IT) {
    case vt::int8::id:
      return FEDP<vt::int8, vt::int32>::eval;
    case vt::uint8::id:
      return FEDP<vt::uint8, vt::int32>::eval;
    case vt::int4::id:
      return FEDP<vt::int4, vt::int32>::eval;
    case vt::uint4::id:
      return FEDP<vt::uint4, vt::int32>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  default:
    std::cout << "Error: unsupported output type: " << OT << "!" << std::endl;
    std::abort();
  }
}

class TensorUnit::Impl {
public:
  Impl(TensorUnit* simobject, const Arch& arch, Core* core)
    : simobject_(simobject)
    , core_(core)
    , arch_(arch)
    , perf_stats_()
  {
    //--
  }

  ~Impl() {
    // Destructor logic if needed
  }

  void reset() {
    perf_stats_ = PerfStats();
  }

  void tick() {
    for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
      auto& input = simobject_->Inputs.at(iw);
      if (input.empty())
        continue;
      auto trace = input.front();
      auto tcu_type = std::get<TcuType>(trace->op_type);
      int delay = 0;
      switch (tcu_type) {
      case TcuType::WMMA:
        delay = 4;
        break;
      default:
        std::abort();
      }
      simobject_->Outputs.at(iw).push(trace, 2 + delay);
      DT(3, simobject_->name() << ": op=" << tcu_type << ", " << *trace);
      input.pop();
    }
  }

//=============================================================================
uint32_t k_passes (uint32_t fmt_s){
  switch (fmt_s) {
    case vt::fp16::id:
      // fp16 相关的预处理
      return 1;
    case vt::fp8::id:
      return 2;
    case vt::fp4::id:
      return 4;
    default:
      std::abort();
  }
  return 1;
}

static PrecisionType map_in_precision(uint32_t fmt_s) {
  switch (fmt_s) {
  case vt::fp16::id:
    return PREC_FP16;
  case vt::fp8::id:
    return PREC_FP8_E4M3;
  case vt::fp4::id:
    return PREC_FP4_E2M1;
  default:
    return PREC_FP16;
  }
}

static PrecisionType map_out_precision(uint32_t fmt_d) {
  switch (fmt_d) {
  case vt::fp16::id:
    return PREC_FP16;
  case vt::fp32::id:
    return PREC_FP32;
  default:
    return PREC_FP16;
  }
}

  void wmma(uint32_t wid,
            uint32_t fmt_s,
            uint32_t fmt_d,
            uint32_t step_m,
            uint32_t step_n,
            const std::vector<reg_data_t>& rs1_data,
            const std::vector<reg_data_t>& rs2_data,
            const std::vector<reg_data_t>& rs3_data,
            std::vector<reg_data_t>& rd_data,
            ExeTraceData* trace_data) {
    __unused(wid);
    __unused(trace_data);

    if (use_open_tensorcore(fmt_s, fmt_d)) {
      static bool open_tensorcore_banner = false;
      if (!open_tensorcore_banner) {
        std::cout << "simx tensor_unit: using open_tensorcore path" << std::endl;
        open_tensorcore_banner = true;
      }
      g_cfg.precisions.clear();
      g_cfg.out_precisions.clear();
      g_cfg.precisions.push_back(map_in_precision(fmt_s));
      g_cfg.out_precisions.push_back(map_out_precision(fmt_d));

      uint16_t c_acc[8][8] = {};
      static constexpr uint32_t tcM = 8;
      static constexpr uint32_t tcN = 8;
      static constexpr uint32_t tcK_words = cfg::tcK;
      static constexpr uint32_t tcN_half = 4;
      uint16_t A[8][8] = {};
      uint16_t B[8][8] = {};
      uint32_t C_zero[8][8] = {0};
      uint32_t k_passes_val = k_passes(fmt_s);
      uint32_t step_k = 0;

      TensorCoreTop tc;
      tc.reset();
      tc.set_jobs = k_passes_val;
      tc.jobs_completed = 0;

      for (uint32_t lane = 0; lane < 32; ++lane) {
        auto c_word = rs3_data.at(lane).u32;
        uint32_t i = lane / 4;
        uint32_t j = (lane % 4) * 2;
        c_acc[i][j + 0] = static_cast<uint16_t>(c_word & 0xffff);
        c_acc[i][j + 1] = static_cast<uint16_t>((c_word >> 16) & 0xffff);
      }

      for (step_k = 0; step_k < k_passes_val; ++step_k) {
        for (uint32_t i = 0; i < tcM; ++i) {
          for (uint32_t k = 0; k < 8; ++k) {
            A[i][k] = 0;
            B[k][i] = 0;
          }
        }
        uint32_t z_begin = step_k * tcK_words / k_passes_val;
        uint32_t z_end = (step_k + 1) * tcK_words / k_passes_val;
        uint32_t lanes_per_word = 2 * k_passes_val;
        for (uint32_t i = 0; i < tcM; ++i) {
          for (uint32_t z = z_begin; z < z_end; ++z) {
            auto a_word = rs1_data.at(i * tcK_words + z).u32;
            uint32_t z_local = z - z_begin;
            for (uint32_t lane = 0; lane < lanes_per_word; ++lane) {
              uint32_t k_idx = (z_local * lanes_per_word + lane) % 8;
              A[i][k_idx] = unpack_word_lane(fmt_s, a_word, lane);
            }
          }
        }
        for (uint32_t j = 0; j < tcN; ++j) {
          for (uint32_t z = z_begin; z < z_end; ++z) {
            uint32_t b_idx = j * tcK_words + z;
            auto b_word = rs2_data.at(b_idx).u32;
            uint32_t z_local = z - z_begin;
            for (uint32_t lane = 0; lane < lanes_per_word; ++lane) {
              uint32_t k_idx = (z_local * lanes_per_word + lane) % 8;
              B[k_idx][j] = unpack_word_lane(fmt_s, b_word, lane);
            }
          }
        }
        tc.load_inputs(A, B, C_zero);
        tc.tick(true);
      }

      tc.load_invalid();

      uint32_t spin_limit = 10000;

      int completed_seen = 0;
      while (spin_limit-- > 0 && tc.jobs_completed < tc.set_jobs) {
        tc.run();

        if (tc.jobs_completed > completed_seen) {
          completed_seen = tc.jobs_completed;
          for (uint32_t i = 0; i < tcM; ++i) {
            for (uint32_t j = 0; j < tcN; ++j) {
              c_acc[i][j] = fp16_add(c_acc[i][j], tc.d_out[i][j]);
            }
          }
        }
      }

      for (uint32_t lane = 0; lane < 32; ++lane) {
        uint32_t i = lane / 4;
        uint32_t j = (lane % 4) * 2;
        uint32_t d0 = c_acc[i][j + 0];
        uint32_t d1 = c_acc[i][j + 1];
        uint32_t packed = d0 | (d1 << 16);
        rd_data.at(lane).u64 = nan_box(packed);
      }
      return;

    }
      
    

    PFN_FEDP fedp = select_FEDP(fmt_s, fmt_d);

    uint32_t a_off = (step_m % cfg::a_sub_blocks) * cfg::a_block_size;
    uint32_t b_off = (step_n % cfg::b_sub_blocks) * cfg::b_block_size;

    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      for (uint32_t j = 0; j < cfg::tcN; ++j) {
        auto a_row = rs1_data.data() + a_off + i * cfg::tcK;
        auto b_col = rs2_data.data() + b_off + j * cfg::tcK;
        auto c_val = rs3_data.at(i * cfg::tcN + j).u32;
        auto d_val = fedp(a_row, b_col, c_val);
        rd_data.at(i * cfg::tcN + j).u64 = nan_box(d_val);

        DTH(3, "FEDP: wid=" << wid << ", i=" << i << ", j=" << j << ", m=" << step_m << ", n=" << step_n << ", a_row={" << std::hex);
        for (uint32_t q = 0; q < cfg::tcK; ++q) {
          if (q) DTN(3, ", ");
          DTN(3, "0x" << a_row[q].u32);
        }
        DTN(3, "}, b_col={");
        for (uint32_t q = 0; q < cfg::tcK; ++q) {
          if (q) DTN(3, ", ");
          DTN(3, "0x" << b_col[q].u32);
        }
        DTN(3, "}, c_val=0x" << c_val << ", d_val=0x" << d_val << std::dec << std::endl);
      }
    }
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }

private:

  TensorUnit*   simobject_;
  Core*         core_;
  Arch          arch_;
  PerfStats     perf_stats_;
};

///////////////////////////////////////////////////////////////////////////////

op_string_t vortex::op_string(TcuType tcu_type, IntrTcuArgs args) {
  switch (tcu_type) {
  case TcuType::WMMA:
    return {"WMMA." + std::string(vt::fmt_string(args.fmt_s)) + "." + std::string(vt::fmt_string(args.fmt_d))
             + "." + std::to_string(args.step_m) + "." + std::to_string(args.step_n), ""};
  default:
    std::abort();
  }
}

///////////////////////////////////////////////////////////////////////////////

TensorUnit::TensorUnit(const SimContext &ctx, const char* name, const Arch& arch, Core* core)
	: SimObject<TensorUnit>(ctx, name)
	, Inputs(ISSUE_WIDTH, this)
	, Outputs(ISSUE_WIDTH, this)
	, impl_(new Impl(this, arch, core))
{}

TensorUnit::~TensorUnit() {
  delete impl_;
}

void TensorUnit::reset() {
  impl_->reset();
}

void TensorUnit::tick() {
  impl_->tick();
}

const TensorUnit::PerfStats &TensorUnit::perf_stats() const {
	return impl_->perf_stats();
}

void TensorUnit::wmma(uint32_t wid,
                      uint32_t fmt_s,
                      uint32_t fmt_d,
                      uint32_t step_m,
                      uint32_t step_n,
                      const std::vector<reg_data_t>& rs1_data,
                      const std::vector<reg_data_t>& rs2_data,
                      const std::vector<reg_data_t>& rs3_data,
                      std::vector<reg_data_t>& rd_data,
                      ExeTraceData* trace_data) {
  impl_->wmma(wid, fmt_s, fmt_d, step_m, step_n, rs1_data, rs2_data, rs3_data, rd_data, trace_data);
}
