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

#include <algorithm>
#include <rvfloats.h>

#include "core.h"
#include "tensor_cfg.h"
#include "open_tensorcore/amem.h"
#include "open_tensorcore/bmem.h"
#include "open_tensorcore/cmem.h"
#include "open_tensorcore/tensor_core_top.h"

using namespace vortex;

namespace vt = vortex::tensor;

namespace {

inline uint64_t nan_box(uint32_t value) {
  return value | 0xffffffff00000000;
}

inline float fp16_to_float(uint16_t input) {
  return bit_cast<float>(rv_htof_s(input, 0, nullptr));
}

inline uint16_t float_to_fp16(float input) {
  return rv_ftoh_s(bit_cast<uint32_t>(input), 0, nullptr);
}

inline uint16_t fp16_add(uint16_t a, uint16_t b) {
  auto xa = rv_htof_s(a, 0, nullptr);
  auto xb = rv_htof_s(b, 0, nullptr);
  auto xc = rv_fadd_s(xa, xb, 0, nullptr);
  return rv_ftoh_s(xc, 0, nullptr);
}

uint16_t unpack_word_lane(uint32_t fmt_ab, uint32_t word, uint32_t lane) {
  switch (fmt_ab) {
  case vt::fp16::id:
    if (lane > 1)
      return 0;
    return (word >> (lane * 16)) & 0xffff;
  case vt::fp8::id:
    if (lane > 3)
      return 0;
    return (word >> (lane * 8)) & 0xff;
  default:
    return 0;
  }
}

bool use_open_tensorcore(uint32_t fmt_ab, uint32_t fmt_c) {
  if (fmt_c != vt::fp16::id)
    return false;
  if constexpr (NUM_THREADS != 32) {
    return false;
  }
  return (fmt_ab == vt::fp8::id || fmt_ab == vt::fp16::id);
}

uint32_t legacy_k_passes(uint32_t fmt_ab) {
  switch (fmt_ab) {
  case vt::fp16::id:
    return 1;
  case vt::fp8::id:
    return 2;
  default:
    std::abort();
  }
}

PrecisionType map_in_precision(uint32_t fmt_ab) {
  switch (fmt_ab) {
  case vt::fp16::id:
    return PREC_FP16;
  case vt::fp8::id:
    return PREC_FP8_E4M3;
  default:
    return PREC_FP16;
  }
}

PrecisionType map_out_precision(uint32_t fmt_c) {
  switch (fmt_c) {
  case vt::fp16::id:
    return PREC_FP16;
  case vt::fp32::id:
    return PREC_FP32;
  default:
    return PREC_FP16;
  }
}

uint32_t ab_packet_count(uint32_t fmt_ab) {
  return (fmt_ab == vt::fp16::id) ? 8 : 4;
}

uint32_t c_packet_count(uint32_t fmt_c) {
  return (fmt_c == vt::fp32::id) ? 16 : 8;
}

uint32_t target_packet_offset(const IntrTcuArgs& args) {
  auto ab_packets = ab_packet_count(args.fmt_ab);
  switch (args.target) {
  case TcuTarget::A:
    return 0;
  case TcuTarget::B:
    return ab_packets;
  case TcuTarget::C:
    return ab_packets * 2;
  default:
    return 0;
  }
}

template <typename PacketT>
std::vector<PacketT> copy_packets(const std::vector<Core::TmemPacket>& packets) {
  std::vector<PacketT> out(packets.size());
  for (size_t i = 0; i < packets.size(); ++i) {
    std::copy_n(packets.at(i).bytes.begin(), packets.at(i).bytes.size(), out.at(i).begin());
  }
  return out;
}

void run_open_tensorcore_primitive(uint32_t fmt_ab,
                                   const uint16_t a_in[8][8],
                                   const uint16_t b_in[8][8],
                                   uint16_t d_out[8][8]) {
  TensorCoreTop tc;
  uint32_t c_zero[8][8] = {};

  g_cfg.precisions.clear();
  g_cfg.out_precisions.clear();
  g_cfg.precisions.push_back(map_in_precision(fmt_ab));
  g_cfg.out_precisions.push_back(PREC_FP16);

  tc.reset();
  tc.load_inputs(a_in, b_in, c_zero);
  tc.tick(true);
  tc.load_invalid();

  uint32_t spin_limit = 10000;
  while (spin_limit-- > 0 && tc.jobs_completed < tc.set_jobs) {
    tc.run();
  }

  for (uint32_t i = 0; i < 8; ++i) {
    for (uint32_t j = 0; j < 8; ++j) {
      d_out[i][j] = tc.d_out[i][j];
    }
  }
}

void pack_fp16_block_to_rd(const uint16_t block[8][8], std::vector<reg_data_t>& rd_data) {
  for (uint32_t lane = 0; lane < 32; ++lane) {
    uint32_t i = lane / 4;
    uint32_t j = (lane % 4) * 2;
    uint32_t d0 = block[i][j + 0];
    uint32_t d1 = block[i][j + 1];
    rd_data.at(lane).u64 = nan_box(d0 | (d1 << 16));
  }
}

uint32_t fedp_fp16_fp16(const reg_data_t* a_row, const reg_data_t* b_col, uint32_t c_val) {
  auto acc = c_val & 0xffff;
  for (uint32_t z = 0; z < vt::wmma_config_t<NUM_THREADS, vt::fp16, vt::fp16>::tcK; ++z) {
    auto a0 = a_row[z].u32 & 0xffff;
    auto a1 = (a_row[z].u32 >> 16) & 0xffff;
    auto b0 = b_col[z].u32 & 0xffff;
    auto b1 = (b_col[z].u32 >> 16) & 0xffff;
    auto x0 = rv_fmul_s(rv_htof_s(a0, 0, nullptr), rv_htof_s(b0, 0, nullptr), 0, nullptr);
    auto x1 = rv_fmul_s(rv_htof_s(a1, 0, nullptr), rv_htof_s(b1, 0, nullptr), 0, nullptr);
    auto sum = rv_fadd_s(x0, x1, 0, nullptr);
    auto accf = rv_htof_s(acc, 0, nullptr);
    acc = rv_ftoh_s(rv_fadd_s(sum, accf, 0, nullptr), 0, nullptr);
  }
  return acc;
}

uint32_t fedp_fp16_fp32(const reg_data_t* a_row, const reg_data_t* b_col, uint32_t c_val) {
  float acc = bit_cast<float>(c_val);
  for (uint32_t z = 0; z < vt::wmma_config_t<NUM_THREADS, vt::fp16, vt::fp32>::tcK; ++z) {
    auto a0 = a_row[z].u32 & 0xffff;
    auto a1 = (a_row[z].u32 >> 16) & 0xffff;
    auto b0 = b_col[z].u32 & 0xffff;
    auto b1 = (b_col[z].u32 >> 16) & 0xffff;
    acc += fp16_to_float(a0) * fp16_to_float(b0);
    acc += fp16_to_float(a1) * fp16_to_float(b1);
  }
  return bit_cast<uint32_t>(acc);
}

void legacy_wmma(uint32_t fmt_ab,
                 uint32_t fmt_c,
                 uint32_t step_m,
                 uint32_t step_n,
                 const std::vector<reg_data_t>& rs1_data,
                 const std::vector<reg_data_t>& rs2_data,
                 const std::vector<reg_data_t>& rs3_data,
                 std::vector<reg_data_t>& rd_data) {
  if (use_open_tensorcore(fmt_ab, fmt_c)) {
    uint16_t a_primitive[8][8] = {};
    uint16_t b_primitive[8][8] = {};
    uint16_t c_block[8][8] = {};
    uint16_t d_block[8][8] = {};
    uint32_t tcK_words = 8 / legacy_k_passes(fmt_ab);

    for (uint32_t lane = 0; lane < 32; ++lane) {
      uint32_t i = lane / 4;
      uint32_t j = (lane % 4) * 2;
      auto word = rs3_data.at(lane).u32;
      c_block[i][j + 0] = word & 0xffff;
      c_block[i][j + 1] = (word >> 16) & 0xffff;
    }

    for (uint32_t step_k = 0; step_k < legacy_k_passes(fmt_ab); ++step_k) {
      uint32_t lanes_per_word = 2 * legacy_k_passes(fmt_ab);
      uint32_t z_begin = step_k * tcK_words / legacy_k_passes(fmt_ab);
      uint32_t z_end = (step_k + 1) * tcK_words / legacy_k_passes(fmt_ab);

      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t z = z_begin; z < z_end; ++z) {
          auto a_word = rs1_data.at(i * tcK_words + z).u32;
          uint32_t z_local = z - z_begin;
          for (uint32_t lane = 0; lane < lanes_per_word; ++lane) {
            uint32_t k_idx = (z_local * lanes_per_word + lane) % 8;
            a_primitive[i][k_idx] = unpack_word_lane(fmt_ab, a_word, lane);
          }
        }
      }
      for (uint32_t j = 0; j < 8; ++j) {
        for (uint32_t z = z_begin; z < z_end; ++z) {
          auto b_word = rs2_data.at(j * tcK_words + z).u32;
          uint32_t z_local = z - z_begin;
          for (uint32_t lane = 0; lane < lanes_per_word; ++lane) {
            uint32_t k_idx = (z_local * lanes_per_word + lane) % 8;
            b_primitive[k_idx][j] = unpack_word_lane(fmt_ab, b_word, lane);
          }
        }
      }

      uint16_t partial[8][8] = {};
      run_open_tensorcore_primitive(fmt_ab, a_primitive, b_primitive, partial);
      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
          c_block[i][j] = fp16_add(c_block[i][j], partial[i][j]);
        }
      }
    }

    pack_fp16_block_to_rd(c_block, rd_data);
    return;
  }

  if (fmt_ab == vt::fp16::id && fmt_c == vt::fp16::id) {
    using cfg = vt::wmma_config_t<NUM_THREADS, vt::fp16, vt::fp16>;
    auto a_off = (step_m % cfg::a_sub_blocks) * cfg::a_block_size;
    auto b_off = (step_n % cfg::b_sub_blocks) * cfg::b_block_size;
    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      for (uint32_t j = 0; j < cfg::tcN; ++j) {
        auto a_row = rs1_data.data() + a_off + i * cfg::tcK;
        auto b_col = rs2_data.data() + b_off + j * cfg::tcK;
        auto c_val = rs3_data.at(i * cfg::tcN + j).u32;
        rd_data.at(i * cfg::tcN + j).u64 = nan_box(fedp_fp16_fp16(a_row, b_col, c_val));
      }
    }
    return;
  }

  if (fmt_ab == vt::fp16::id && fmt_c == vt::fp32::id) {
    using cfg = vt::wmma_config_t<NUM_THREADS, vt::fp16, vt::fp32>;
    auto a_off = (step_m % cfg::a_sub_blocks) * cfg::a_block_size;
    auto b_off = (step_n % cfg::b_sub_blocks) * cfg::b_block_size;
    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      for (uint32_t j = 0; j < cfg::tcN; ++j) {
        auto a_row = rs1_data.data() + a_off + i * cfg::tcK;
        auto b_col = rs2_data.data() + b_off + j * cfg::tcK;
        auto c_val = rs3_data.at(i * cfg::tcN + j).u32;
        rd_data.at(i * cfg::tcN + j).u64 = nan_box(fedp_fp16_fp32(a_row, b_col, c_val));
      }
    }
    return;
  }

  std::abort();
}

struct WarpTensorState {
  AMem amem;
  BMem bmem;
  CMem cmem;

  void reset() {
    amem.reset();
    bmem.reset();
    cmem.reset();
  }

  bool ready(bool ws) const {
    return amem.valid() && cmem.valid() && (ws || bmem.valid());
  }
};

} // namespace

class TensorUnit::Impl {
public:
  Impl(TensorUnit* simobject, const Arch& arch, Core* core)
    : simobject_(simobject)
    , core_(core)
    , arch_(arch)
    , perf_stats_()
    , warp_state_(arch.num_warps()) {
    for (auto& state : warp_state_) {
      state.reset();
    }
  }

  void reset() {
    perf_stats_ = PerfStats();
    for (auto& state : warp_state_) {
      state.reset();
    }
  }

  void tick() {
    for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
      auto& input = simobject_->Inputs.at(iw);
      if (input.empty())
        continue;
      auto trace = input.front();
      auto tcu_type = std::get<TcuType>(trace->op_type);
      int delay = 1;
      switch (tcu_type) {
      case TcuType::MMA_LOAD:
      case TcuType::MMA_STORE:
        delay = 2;
        break;
      case TcuType::WMMA:
        delay = 4;
        break;
      default:
        delay = 1;
        break;
      }
      simobject_->Outputs.at(iw).push(trace, 2 + delay);
      DT(3, simobject_->name() << ": op=" << tcu_type << ", " << *trace);
      input.pop();
    }
  }

  void mma_load(uint32_t wid, uint32_t handle, IntrTcuArgs args, ExeTraceData* trace_data) {
    auto packets_needed = (args.target == TcuTarget::C) ? c_packet_count(args.fmt_c) : ab_packet_count(args.fmt_ab);
    auto packet_offset = target_packet_offset(args);
    std::vector<Core::TmemPacket> packets(packets_needed);
    for (uint32_t i = 0; i < packets_needed; ++i) {
      if (!core_->tmem_read_packet(handle, packet_offset + i, &packets.at(i))) {
        std::abort();
      }
    }

    auto& state = warp_state_.at(wid);
    switch (args.target) {
    case TcuTarget::A: {
      auto packet_bytes = copy_packets<AMem::packet_t>(packets);
      if (!state.amem.fill_tile(args.fmt_ab, packet_bytes)) {
        std::abort();
      }
    } break;
    case TcuTarget::B: {
      auto packet_bytes = copy_packets<BMem::packet_t>(packets);
      if (!state.bmem.fill_tile(args.fmt_ab, packet_bytes)) {
        std::abort();
      }
    } break;
    case TcuTarget::C: {
      auto packet_bytes = copy_packets<CMem::packet_t>(packets);
      if (!state.cmem.fill_tile(args.fmt_c, packet_bytes)) {
        std::abort();
      }
    } break;
    default:
      std::abort();
    }
    if (trace_data) {
      trace_data->rd_write = false;
    }
  }

  void mma_store(uint32_t wid, uint32_t handle, IntrTcuArgs args, ExeTraceData* trace_data) {
    auto& state = warp_state_.at(wid);
    std::vector<CMem::packet_t> packets;
    if (!state.cmem.dump_tile(args.fmt_c, &packets)) {
      std::abort();
    }
    auto packet_offset = 0u;
    for (uint32_t i = 0; i < packets.size(); ++i) {
      Core::TmemPacket packet;
      std::copy_n(packets.at(i).begin(), packets.at(i).size(), packet.bytes.begin());
      if (!core_->tmem_write_packet(handle, packet_offset + i, packet)) {
        std::abort();
      }
    }
    if (trace_data) {
      trace_data->rd_write = false;
    }
  }

  bool try_storage_wmma(uint32_t wid,
                        uint32_t fmt_ab,
                        uint32_t fmt_c,
                        uint32_t step_m,
                        uint32_t step_n,
                        uint32_t step_k,
                        std::vector<reg_data_t>& rd_data,
                        ExeTraceData* trace_data) {
    auto map_storage_steps = [&](auto cfg_tag, uint32_t* storage_m, uint32_t* storage_n, uint32_t* storage_k) {
      using cfg = decltype(cfg_tag);
      static_assert(8 % cfg::tcM == 0, "storage mapping expects tcM to divide primitive M");
      static_assert(8 % cfg::tcN == 0, "storage mapping expects tcN to divide primitive N");
      static_assert(8 % cfg::tcK == 0, "storage mapping expects tcK to divide primitive K");
      constexpr uint32_t m_factor = 8 / cfg::tcM;
      constexpr uint32_t n_factor = 8 / cfg::tcN;
      constexpr uint32_t k_factor = 8 / cfg::tcK;
      if ((step_m % m_factor) != 0 || (step_n % n_factor) != 0 || (step_k % k_factor) != 0) {
        return false;
      }
      *storage_m = step_m / m_factor;
      *storage_n = step_n / n_factor;
      *storage_k = step_k / k_factor;
      return (*storage_m < 2) && (*storage_n < 2) && (*storage_k < 2);
    };

    auto& state = warp_state_.at(wid);
    if (!state.ready(false))
      return false;

    uint32_t storage_m = 0;
    uint32_t storage_n = 0;
    uint32_t storage_k = 0;
    bool step_active = false;
    if (fmt_c == vt::fp16::id) {
      if (fmt_ab == vt::fp8::id) {
        step_active = map_storage_steps(vt::wmma_config_t<NUM_THREADS, vt::fp8, vt::fp16>{}, &storage_m, &storage_n, &storage_k);
      } else {
        step_active = map_storage_steps(vt::wmma_config_t<NUM_THREADS, vt::fp16, vt::fp16>{}, &storage_m, &storage_n, &storage_k);
      }
    } else {
      if (fmt_ab == vt::fp8::id) {
        step_active = map_storage_steps(vt::wmma_config_t<NUM_THREADS, vt::fp8, vt::fp32>{}, &storage_m, &storage_n, &storage_k);
      } else {
        step_active = map_storage_steps(vt::wmma_config_t<NUM_THREADS, vt::fp16, vt::fp32>{}, &storage_m, &storage_n, &storage_k);
      }
    }
    if (!step_active) {
      if (trace_data) {
        trace_data->rd_write = false;
      }
      return true;
    }

    uint16_t a_block[8][8] = {};
    uint16_t b_block[8][8] = {};
    state.amem.read_primitive(storage_m, storage_k, a_block);
    state.bmem.read_primitive(storage_k, storage_n, b_block);

    if (fmt_c == vt::fp16::id) {
      uint16_t c_block[8][8] = {};
      uint16_t partial[8][8] = {};
      state.cmem.load_block_fp16(storage_m, storage_n, c_block);
      run_open_tensorcore_primitive(fmt_ab, a_block, b_block, partial);
      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
          c_block[i][j] = fp16_add(c_block[i][j], partial[i][j]);
        }
      }
      state.cmem.store_block_fp16(storage_m, storage_n, c_block);
      if (trace_data) {
        trace_data->rd_write = false;
      }
      return true;
    }

    if (fmt_c == vt::fp32::id) {
      float c_block[8][8] = {};
      state.cmem.load_block_fp32(storage_m, storage_n, c_block);
      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
          float acc = c_block[i][j];
          for (uint32_t k = 0; k < 8; ++k) {
            acc += static_cast<float>(fp9_to_double(a_block[i][k]) * fp9_to_double(b_block[k][j]));
          }
          c_block[i][j] = acc;
        }
      }
      state.cmem.store_block_fp32(storage_m, storage_n, c_block);
      if (trace_data) {
        trace_data->rd_write = false;
      }
      return true;
    }

    return false;
  }

  void wmma(uint32_t wid,
            uint32_t fmt_ab,
            uint32_t fmt_c,
            uint32_t step_m,
            uint32_t step_n,
            uint32_t step_k,
            const std::vector<reg_data_t>& rs1_data,
            const std::vector<reg_data_t>& rs2_data,
            const std::vector<reg_data_t>& rs3_data,
            std::vector<reg_data_t>& rd_data,
            ExeTraceData* trace_data) {
    if (try_storage_wmma(wid, fmt_ab, fmt_c, step_m, step_n, step_k, rd_data, trace_data)) {
      return;
    }

    if (trace_data) {
      trace_data->rd_write = true;
    }
    legacy_wmma(fmt_ab, fmt_c, step_m, step_n, rs1_data, rs2_data, rs3_data, rd_data);
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }

private:
  TensorUnit* simobject_;
  Core*       core_;
  Arch        arch_;
  PerfStats   perf_stats_;
  std::vector<WarpTensorState> warp_state_;
};

op_string_t vortex::op_string(TcuType tcu_type, IntrTcuArgs args) {
  switch (tcu_type) {
  case TcuType::TMEM_ALLOC:
    return {"TMEM_ALLOC." + std::to_string(args.bank_span), ""};
  case TcuType::TMEM_FREE:
    return {"TMEM_FREE", ""};
  case TcuType::TMA_LOAD:
    return {"TMA_LOAD" + std::string(args.transpose_b ? ".TB" : ""), ""};
  case TcuType::TMA_STORE:
    return {"TMA_STORE", ""};
  case TcuType::MMA_LOAD:
    return {"MMA_LOAD." + std::string(vt::fmt_string(args.fmt_ab)) + "." + std::string(vt::fmt_string(args.fmt_c))
          + "." + std::to_string(static_cast<uint32_t>(args.target)), ""};
  case TcuType::MMA_STORE:
    return {"MMA_STORE." + std::string(vt::fmt_string(args.fmt_c)), ""};
  case TcuType::TMA_WAIT:
    return {"TMA_WAIT", ""};
  case TcuType::WMMA:
    return {"WMMA." + std::string(vt::fmt_string(args.fmt_ab)) + "." + std::string(vt::fmt_string(args.fmt_c))
          + "." + std::to_string(args.step_m) + "." + std::to_string(args.step_n) + "." + std::to_string(args.step_k), ""};
  default:
    std::abort();
  }
}

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

void TensorUnit::mma_load(uint32_t wid,
                          uint32_t handle,
                          IntrTcuArgs args,
                          ExeTraceData* trace_data) {
  impl_->mma_load(wid, handle, args, trace_data);
}

void TensorUnit::mma_store(uint32_t wid,
                           uint32_t handle,
                           IntrTcuArgs args,
                           ExeTraceData* trace_data) {
  impl_->mma_store(wid, handle, args, trace_data);
}

void TensorUnit::wmma(uint32_t wid,
                      uint32_t fmt_ab,
                      uint32_t fmt_c,
                      uint32_t step_m,
                      uint32_t step_n,
                      uint32_t step_k,
                      const std::vector<reg_data_t>& rs1_data,
                      const std::vector<reg_data_t>& rs2_data,
                      const std::vector<reg_data_t>& rs3_data,
                      std::vector<reg_data_t>& rd_data,
                      ExeTraceData* trace_data) {
  impl_->wmma(wid, fmt_ab, fmt_c, step_m, step_n, step_k, rs1_data, rs2_data, rs3_data, rd_data, trace_data);
}

const TensorUnit::PerfStats& TensorUnit::perf_stats() const {
  return impl_->perf_stats();
}
