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
#include <deque>
#include <unordered_map>
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

bool use_open_tensorcore(uint32_t fmt_a, uint32_t fmt_b, uint32_t fmt_c) {
  if constexpr (NUM_THREADS != 32) {
    return false;
  }
  if (fmt_c != vt::fp16::id && fmt_c != vt::fp32::id)
    return false;
  bool supported_a = (fmt_a == vt::fp8::id || fmt_a == vt::fp16::id);
  bool supported_b = (fmt_b == vt::fp8::id || fmt_b == vt::fp16::id);
  return supported_a && supported_b;
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
  auto fmt_a = args.fmt_a ? args.fmt_a : args.fmt_ab;
  auto fmt_b = args.fmt_b ? args.fmt_b : args.fmt_ab;
  auto a_packets = ab_packet_count(fmt_a);
  auto b_packets = ab_packet_count(fmt_b);
  switch (args.target) {
  case TcuTarget::A:
    return 0;
  case TcuTarget::B:
    return a_packets;
  case TcuTarget::C:
    return a_packets + b_packets;
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

void configure_open_tensorcore_precision(uint32_t fmt_c) {
  g_cfg.precisions.clear();
  g_cfg.out_precisions.clear();
  g_cfg.precisions.push_back(PREC_FP9);
  g_cfg.out_precisions.push_back(map_out_precision(fmt_c));
}

void init_open_tensorcore_job(TensorCoreTop* tc,
                              uint32_t fmt_c,
                              const uint16_t a_in[8][8],
                              const uint16_t b_in[8][8],
                              const uint32_t c_in[8][8]) {
  configure_open_tensorcore_precision(fmt_c);
  tc->reset();
  tc->load_inputs(a_in, b_in, c_in);
  tc->tick(true);
  tc->load_invalid();
}

bool step_open_tensorcore_job(TensorCoreTop* tc, uint32_t fmt_c) {
  configure_open_tensorcore_precision(fmt_c);
  return tc->run();
}

void run_open_tensorcore_primitive_blocking(uint32_t fmt_a,
                                            uint32_t fmt_b,
                                            uint32_t fmt_c,
                                            const uint16_t a_in[8][8],
                                            const uint16_t b_in[8][8],
                                            uint32_t d_out[8][8],
                                            const uint32_t (*c_in)[8] = nullptr) {
  TensorCoreTop tc;
  uint32_t c_raw[8][8] = {};
  (void)fmt_a;
  (void)fmt_b;
  if (c_in != nullptr) {
    for (uint32_t i = 0; i < 8; ++i) {
      for (uint32_t j = 0; j < 8; ++j) {
        c_raw[i][j] = c_in[i][j];
      }
    }
  }

  init_open_tensorcore_job(&tc, fmt_c, a_in, b_in, c_raw);

  bool completed = false;
  uint32_t spin_limit = 10000;
  while (spin_limit-- > 0) {
    if (step_open_tensorcore_job(&tc, fmt_c)) {
      completed = true;
      break;
    }
  }
  if (!completed) {
    std::abort();
  }

  for (uint32_t i = 0; i < 8; ++i) {
    for (uint32_t j = 0; j < 8; ++j) {
      d_out[i][j] = tc.d_out[i][j];
    }
  }
}

void encode_c_block_fp16(const uint16_t in[8][8], uint32_t out[8][8]) {
  for (uint32_t i = 0; i < 8; ++i) {
    for (uint32_t j = 0; j < 8; ++j) {
      out[i][j] = in[i][j];
    }
  }
}

void decode_c_block_fp16(const uint32_t in[8][8], uint16_t out[8][8]) {
  for (uint32_t i = 0; i < 8; ++i) {
    for (uint32_t j = 0; j < 8; ++j) {
      out[i][j] = in[i][j] & 0xffff;
    }
  }
}

void encode_c_block_fp32(const float in[8][8], uint32_t out[8][8]) {
  for (uint32_t i = 0; i < 8; ++i) {
    for (uint32_t j = 0; j < 8; ++j) {
      out[i][j] = bit_cast<uint32_t>(in[i][j]);
    }
  }
}

void decode_c_block_fp32(const uint32_t in[8][8], float out[8][8]) {
  for (uint32_t i = 0; i < 8; ++i) {
    for (uint32_t j = 0; j < 8; ++j) {
      out[i][j] = bit_cast<float>(in[i][j]);
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

void pack_fp32_block_to_rd(const uint32_t block[8][8], std::vector<reg_data_t>& rd_data) {
  for (uint32_t lane = 0; lane < 32; ++lane) {
    uint32_t i = lane / 4;
    uint32_t j = lane % 4;
    rd_data.at(lane).u64 = nan_box(block[i][j]);
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

void legacy_wmma(uint32_t fmt_a,
                 uint32_t fmt_b,
                 uint32_t fmt_c,
                 uint32_t step_m,
                 uint32_t step_n,
                 const std::vector<reg_data_t>& rs1_data,
                 const std::vector<reg_data_t>& rs2_data,
                 const std::vector<reg_data_t>& rs3_data,
                 std::vector<reg_data_t>& rd_data) {
  if (fmt_a != fmt_b) {
    std::abort();
  }

  auto fmt_ab = fmt_a;
  if (use_open_tensorcore(fmt_a, fmt_b, fmt_c)) {
    uint16_t a_primitive[8][8] = {};
    uint16_t b_primitive[8][8] = {};
    uint32_t c_raw[8][8] = {};
    uint32_t d_raw[8][8] = {};
    uint32_t tcK_words = 8 / legacy_k_passes(fmt_ab);

    if (fmt_c == vt::fp16::id) {
      uint16_t c_block[8][8] = {};
      for (uint32_t lane = 0; lane < 32; ++lane) {
        uint32_t i = lane / 4;
        uint32_t j = (lane % 4) * 2;
        auto word = rs3_data.at(lane).u32;
        c_block[i][j + 0] = word & 0xffff;
        c_block[i][j + 1] = (word >> 16) & 0xffff;
      }
      encode_c_block_fp16(c_block, c_raw);
    } else {
      for (uint32_t lane = 0; lane < 32; ++lane) {
        uint32_t i = lane / 4;
        uint32_t j = lane % 4;
        c_raw[i][j] = rs3_data.at(lane).u32;
      }
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

      run_open_tensorcore_primitive_blocking(fmt_a, fmt_b, fmt_c, a_primitive, b_primitive, d_raw, c_raw);
      std::copy(&d_raw[0][0], &d_raw[0][0] + 64, &c_raw[0][0]);
    }

    if (fmt_c == vt::fp16::id) {
      uint16_t c_block[8][8] = {};
      decode_c_block_fp16(c_raw, c_block);
      pack_fp16_block_to_rd(c_block, rd_data);
    } else {
      pack_fp32_block_to_rd(c_raw, rd_data);
    }
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

} // namespace

class TensorUnit::Impl {
public:
  static constexpr uint32_t kNumOperandSlots = 2;
  static constexpr uint32_t kWmmaPrimitiveCount = 8;
  static constexpr uint32_t kSubtilesPerTile = 4;
  static constexpr uint32_t kPrimitiveDim = 8;
  static constexpr uint32_t kWmmaIssueBurst = 4;

  struct OperandSlot {
    AMem amem;
    BMem bmem;
    CMem cmem;
    std::array<std::array<std::array<uint32_t, kPrimitiveDim>, kPrimitiveDim>, kSubtilesPerTile> accum_fp22 = {};
    uint32_t owner_wid = 0;
    uint32_t descriptor = 0xffffffffu;
    uint32_t fmt_a = 0;
    uint32_t fmt_b = 0;
    uint32_t fmt_c = 0;
    uint32_t wmma_async_id = 0;
    uint32_t store_async_id = 0;
    uint8_t retired_wmma_uops = 0;
    std::array<uint8_t, kSubtilesPerTile> retired_c_subtiles = {};
    bool valid = false;
    bool busy = false;
    bool a_ready = false;
    bool b_ready = false;
    bool c_ready = false;
    bool a_pending = false;
    bool b_pending = false;
    bool c_pending = false;
    bool wmma_pending = false;
    bool cmem_final_valid = false;
    bool store_pending = false;

    void reset() {
      amem.reset();
      bmem.reset();
      cmem.reset();
      for (auto& subtile : accum_fp22) {
        for (auto& row : subtile) {
          row.fill(0);
        }
      }
      owner_wid = 0;
      descriptor = 0xffffffffu;
      fmt_a = 0;
      fmt_b = 0;
      fmt_c = 0;
      wmma_async_id = 0;
      store_async_id = 0;
      retired_wmma_uops = 0;
      retired_c_subtiles.fill(0);
      valid = false;
      busy = false;
      a_ready = false;
      b_ready = false;
      c_ready = false;
      a_pending = false;
      b_pending = false;
      c_pending = false;
      wmma_pending = false;
      cmem_final_valid = false;
      store_pending = false;
    }
  };

  struct PendingWmmaJob {
    uint32_t wid = 0;
    uint32_t slot_id = 0;
    uint32_t fmt_a = 0;
    uint32_t fmt_b = 0;
    uint32_t fmt_c = 0;
    uint32_t async_id = 0;
    uint32_t next_uop = 0;
  };

  struct MemUop {
    enum class Kind : uint8_t {
      FillA = 0,
      FillB,
      FillC,
      StoreC,
    };

    Kind kind = Kind::FillA;
    uint32_t wid = 0;
    uint32_t slot_id = 0;
    uint32_t handle = 0;
    uint32_t async_id = 0;
    bool separate_handle = false;
  };

  Impl(TensorUnit* simobject, const Arch& arch, Core* core)
    : simobject_(simobject)
    , core_(core)
    , arch_(arch)
    , perf_stats_() {
    reset();
  }

  void reset() {
    perf_stats_ = PerfStats();
    for (auto& slot : operand_slots_) {
      slot.reset();
    }
    mem_uops_.clear();
    pending_load_uops_.clear();
    pending_wmma_jobs_.clear();
    active_wmma_job_ = {};
    active_wmma_job_valid_ = false;
    active_wmma_job_budget_ = 0;
    tensorcore_.reset();
    tensorcore_fmt_c_ = vt::fp16::id;
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

    service_mem_uop();
    dispatch_compute_uop();
    tick_tensorcore();
  }

  void mma_load(uint32_t wid, uint32_t handle, IntrTcuArgs args, ExeTraceData* trace_data) {
    if (trace_data) {
      trace_data->rd_write = false;
      trace_data->retry = false;
    }
    if (args.macro_op) {
      enqueue_async_mma_load(wid, handle, args, trace_data);
      return;
    }
    auto slot_id = acquire_slot_for_wid(wid);
    if (slot_id < 0) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    auto& slot = operand_slots_.at(slot_id);
    ensure_slot_matches(slot, wid, args);
    execute_fill(slot, handle, args.target, false);
    mark_target_ready(slot, args.target);
  }

  void mma_store(uint32_t wid, uint32_t handle, IntrTcuArgs args, ExeTraceData* trace_data) {
    if (trace_data) {
      trace_data->rd_write = false;
      trace_data->retry = false;
    }
    if (args.macro_op) {
      enqueue_async_mma_store(wid, handle, args, trace_data);
      return;
    }
    auto slot_id = find_slot_by_owner(wid);
    if (slot_id < 0) {
      std::abort();
    }
    auto& slot = operand_slots_.at(slot_id);
    if (slot.wmma_pending || !slot.cmem_final_valid) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    std::vector<CMem::packet_t> packets;
    if (!slot.cmem.dump_tile(args.fmt_c, &packets)) {
      std::abort();
    }
    for (uint32_t i = 0; i < packets.size(); ++i) {
      Core::TmemPacket packet;
      std::copy_n(packets.at(i).begin(), packets.at(i).size(), packet.bytes.begin());
      if (!core_->tmem_write_packet(handle, i, packet)) {
        std::abort();
      }
    }
  }

  void wmma(uint32_t wid,
            IntrTcuArgs args,
            const std::vector<reg_data_t>& rs1_data,
            const std::vector<reg_data_t>& rs2_data,
            const std::vector<reg_data_t>& rs3_data,
            std::vector<reg_data_t>& rd_data,
            ExeTraceData* trace_data) {
    auto fmt_a = args.fmt_a ? args.fmt_a : args.fmt_ab;
    auto fmt_b = args.fmt_b ? args.fmt_b : args.fmt_ab;
    if (trace_data) {
      trace_data->rd_write = false;
      trace_data->retry = false;
    }

    if (args.macro_op) {
      enqueue_async_wmma(wid, fmt_a, fmt_b, args.fmt_c, trace_data);
      return;
    }

    if (trace_data) {
      trace_data->rd_write = true;
      trace_data->retry = false;
    }
    legacy_wmma(fmt_a, fmt_b, args.fmt_c, args.step_m, args.step_n, rs1_data, rs2_data, rs3_data, rd_data);
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }

  uint32_t scheduler_score(uint32_t wid, TcuType tcu_type, IntrTcuArgs args) const {
    if (!args.macro_op) {
      return 1;
    }

    switch (tcu_type) {
    case TcuType::WMMA: {
      auto fmt_a = args.fmt_a ? args.fmt_a : args.fmt_ab;
      auto fmt_b = args.fmt_b ? args.fmt_b : args.fmt_ab;
      if (!use_open_tensorcore(fmt_a, fmt_b, args.fmt_c)) {
        return 0;
      }
      auto slot_id = find_slot_by_owner(wid);
      if (slot_id < 0) {
        return 0;
      }
      const auto& slot = operand_slots_.at(slot_id);
      return (slot.a_ready && slot.b_ready && slot.c_ready && !slot.wmma_pending && !slot.store_pending) ? 5 : 0;
    }
    case TcuType::MMA_LOAD:
      return (acquire_slot_for_wid(wid) >= 0) ? 4 : 0;
    case TcuType::MMA_STORE: {
      auto slot_id = find_slot_by_owner(wid);
      if (slot_id < 0) {
        return 0;
      }
      const auto& slot = operand_slots_.at(slot_id);
      return (!slot.wmma_pending && slot.cmem_final_valid && !slot.store_pending) ? 3 : 0;
    }
    default:
      return 1;
    }
  }

private:
  static uint32_t add_fp22_raw(uint32_t a, uint32_t b) {
    auto s1 = fadd_s1(a, b, 8, 14, 14, g_cfg.rm);
    return fadd_s2(s1, 8, 14);
  }

  int32_t find_slot_by_owner(uint32_t wid) const {
    for (uint32_t i = 0; i < operand_slots_.size(); ++i) {
      const auto& slot = operand_slots_.at(i);
      if (slot.valid && slot.owner_wid == wid) {
        return i;
      }
    }
    return -1;
  }

  int32_t find_free_slot() const {
    for (uint32_t i = 0; i < operand_slots_.size(); ++i) {
      if (!operand_slots_.at(i).valid) {
        return i;
      }
    }
    return -1;
  }

  int32_t acquire_slot_for_wid(uint32_t wid) const {
    auto owned = find_slot_by_owner(wid);
    if (owned >= 0) {
      const auto& slot = operand_slots_.at(owned);
      if (slot.wmma_pending || slot.store_pending) {
        return -1;
      }
      return owned;
    }
    return find_free_slot();
  }

  void release_slot(uint32_t slot_id) {
    operand_slots_.at(slot_id).reset();
  }

  static bool slot_has_pending_work(const OperandSlot& slot) {
    return slot.a_pending || slot.b_pending || slot.c_pending || slot.wmma_pending || slot.store_pending;
  }

  void init_slot_for_descriptor(OperandSlot& slot, uint32_t wid, const IntrTcuArgs& args) {
    slot.reset();
    slot.valid = true;
    slot.owner_wid = wid;
    slot.descriptor = args.descriptor;
    slot.fmt_a = args.fmt_a ? args.fmt_a : args.fmt_ab;
    slot.fmt_b = args.fmt_b ? args.fmt_b : args.fmt_ab;
    slot.fmt_c = args.fmt_c;
  }

  void ensure_slot_matches(OperandSlot& slot, uint32_t wid, const IntrTcuArgs& args) {
    if (!slot.valid) {
      init_slot_for_descriptor(slot, wid, args);
      return;
    }
    if (slot.owner_wid != wid || slot.descriptor != args.descriptor) {
      std::abort();
    }
  }

  void mark_target_pending(OperandSlot& slot, TcuTarget target, bool pending) {
    switch (target) {
    case TcuTarget::A:
      slot.a_pending = pending;
      if (pending) {
        slot.a_ready = false;
      }
      break;
    case TcuTarget::B:
      slot.b_pending = pending;
      if (pending) {
        slot.b_ready = false;
      }
      break;
    case TcuTarget::C:
      slot.c_pending = pending;
      if (pending) {
        slot.c_ready = false;
        slot.cmem_final_valid = false;
      }
      break;
    default:
      std::abort();
    }
    slot.busy = slot_has_pending_work(slot);
  }

  void mark_target_ready(OperandSlot& slot, TcuTarget target) {
    switch (target) {
    case TcuTarget::A:
      slot.a_pending = false;
      slot.a_ready = true;
      break;
    case TcuTarget::B:
      slot.b_pending = false;
      slot.b_ready = true;
      break;
    case TcuTarget::C:
      slot.c_pending = false;
      slot.c_ready = true;
      slot.cmem_final_valid = true;
      break;
    default:
      std::abort();
    }
    slot.busy = slot_has_pending_work(slot);
  }

  void execute_fill(OperandSlot& slot, uint32_t handle, TcuTarget target, bool separate_handle) {
    IntrTcuArgs resolved_args{};
    resolved_args.descriptor = slot.descriptor;
    resolved_args.fmt_a = slot.fmt_a;
    resolved_args.fmt_b = slot.fmt_b;
    resolved_args.fmt_ab = (slot.fmt_a == slot.fmt_b) ? slot.fmt_a : 0;
    resolved_args.fmt_c = slot.fmt_c;
    resolved_args.target = target;

    auto packets_needed = (target == TcuTarget::C)
                        ? c_packet_count(slot.fmt_c)
                        : ab_packet_count(target == TcuTarget::A ? slot.fmt_a : slot.fmt_b);
    auto packet_offset = separate_handle ? 0 : target_packet_offset(resolved_args);
    std::vector<Core::TmemPacket> packets(packets_needed);
    for (uint32_t i = 0; i < packets_needed; ++i) {
      if (!core_->tmem_read_packet(handle, packet_offset + i, &packets.at(i))) {
        std::abort();
      }
    }

    switch (target) {
    case TcuTarget::A: {
      auto packet_bytes = copy_packets<AMem::packet_t>(packets);
      if (!slot.amem.fill_tile(slot.fmt_a, packet_bytes)) {
        std::abort();
      }
    } break;
    case TcuTarget::B: {
      auto packet_bytes = copy_packets<BMem::packet_t>(packets);
      if (!slot.bmem.fill_tile(slot.fmt_b, packet_bytes)) {
        std::abort();
      }
    } break;
    case TcuTarget::C: {
      auto packet_bytes = copy_packets<CMem::packet_t>(packets);
      if (!slot.cmem.fill_tile(slot.fmt_c, packet_bytes)) {
        std::abort();
      }
    } break;
    default:
      std::abort();
    }
  }

  void init_slot_accumulators(OperandSlot& slot) {
    auto out_prec = map_out_precision(slot.fmt_c);
    for (uint32_t subtile = 0; subtile < kSubtilesPerTile; ++subtile) {
      uint32_t storage_m = subtile / 2;
      uint32_t storage_n = subtile % 2;
      if (slot.fmt_c == vt::fp16::id) {
        uint16_t c_block[kPrimitiveDim][kPrimitiveDim] = {};
        slot.cmem.load_block_fp16(storage_m, storage_n, c_block);
        for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
          for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
            slot.accum_fp22.at(subtile).at(i).at(j) = convert_c_to_fp22(c_block[i][j], out_prec);
          }
        }
      } else if (slot.fmt_c == vt::fp32::id) {
        float c_block[kPrimitiveDim][kPrimitiveDim] = {};
        slot.cmem.load_block_fp32(storage_m, storage_n, c_block);
        for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
          for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
            slot.accum_fp22.at(subtile).at(i).at(j) = convert_c_to_fp22(bit_cast<uint32_t>(c_block[i][j]), out_prec);
          }
        }
      } else {
        std::abort();
      }
    }
  }

  void flush_slot_subtile(OperandSlot& slot, uint32_t subtile_id) {
    uint32_t storage_m = subtile_id / 2;
    uint32_t storage_n = subtile_id % 2;
    if (slot.fmt_c == vt::fp16::id) {
      uint16_t c_block[kPrimitiveDim][kPrimitiveDim] = {};
      for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
        for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
          c_block[i][j] = fp22_to_fp16(slot.accum_fp22.at(subtile_id).at(i).at(j));
        }
      }
      slot.cmem.store_block_fp16(storage_m, storage_n, c_block);
      return;
    }
    if (slot.fmt_c == vt::fp32::id) {
      float c_block[kPrimitiveDim][kPrimitiveDim] = {};
      for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
        for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
          c_block[i][j] = bit_cast<float>(fp22_to_fp32(slot.accum_fp22.at(subtile_id).at(i).at(j)));
        }
      }
      slot.cmem.store_block_fp32(storage_m, storage_n, c_block);
      return;
    }
    std::abort();
  }

  void enqueue_async_mma_load(uint32_t wid, uint32_t handle, const IntrTcuArgs& args, ExeTraceData* trace_data) {
    auto slot_id = acquire_slot_for_wid(wid);
    if (slot_id < 0) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    auto& slot = operand_slots_.at(slot_id);
    if (args.target == TcuTarget::None) {
      init_slot_for_descriptor(slot, wid, args);
    } else {
      ensure_slot_matches(slot, wid, args);
    }

    auto async_id = core_->mma_load_async_issue(wid, handle, args.descriptor);
    uint32_t num_uops = 0;
    auto push_fill = [&](MemUop::Kind kind, TcuTarget target, bool separate_handle) {
      mark_target_pending(slot, target, true);
      mem_uops_.push_back(MemUop{kind, wid, static_cast<uint32_t>(slot_id), handle, async_id, separate_handle});
      ++num_uops;
    };

    if (args.target == TcuTarget::None) {
      push_fill(MemUop::Kind::FillA, TcuTarget::A, false);
      push_fill(MemUop::Kind::FillB, TcuTarget::B, false);
      push_fill(MemUop::Kind::FillC, TcuTarget::C, false);
    } else {
      switch (args.target) {
      case TcuTarget::A:
        push_fill(MemUop::Kind::FillA, TcuTarget::A, true);
        break;
      case TcuTarget::B:
        push_fill(MemUop::Kind::FillB, TcuTarget::B, true);
        break;
      case TcuTarget::C:
        push_fill(MemUop::Kind::FillC, TcuTarget::C, true);
        break;
      default:
        std::abort();
      }
    }

    pending_load_uops_[async_id] = num_uops;
  }

  void enqueue_async_mma_store(uint32_t wid, uint32_t handle, const IntrTcuArgs& args, ExeTraceData* trace_data) {
    auto slot_id = find_slot_by_owner(wid);
    if (slot_id < 0) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    auto& slot = operand_slots_.at(slot_id);
    if (args.target != TcuTarget::None && args.target != TcuTarget::C) {
      std::abort();
    }
    if (slot.wmma_pending || !slot.cmem_final_valid || slot.store_pending) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    slot.store_pending = true;
    slot.busy = true;
    slot.store_async_id = core_->mma_store_async_issue(wid, handle, args.descriptor);
    mem_uops_.push_back(MemUop{MemUop::Kind::StoreC, wid, static_cast<uint32_t>(slot_id), handle, slot.store_async_id, args.target != TcuTarget::None});
  }

  void enqueue_async_wmma(uint32_t wid,
                          uint32_t fmt_a,
                          uint32_t fmt_b,
                          uint32_t fmt_c,
                          ExeTraceData* trace_data) {
    if (!use_open_tensorcore(fmt_a, fmt_b, fmt_c)) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    auto slot_id = find_slot_by_owner(wid);
    if (slot_id < 0) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    auto& slot = operand_slots_.at(slot_id);
    if (!(slot.a_ready && slot.b_ready && slot.c_ready) || slot.wmma_pending || slot.store_pending) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }

    init_slot_accumulators(slot);
    slot.wmma_pending = true;
    slot.busy = true;
    slot.cmem_final_valid = false;
    slot.retired_wmma_uops = 0;
    slot.retired_c_subtiles.fill(0);
    slot.wmma_async_id = core_->wmma_async_issue(wid);
    pending_wmma_jobs_.push_back(PendingWmmaJob{wid, static_cast<uint32_t>(slot_id), fmt_a, fmt_b, fmt_c, slot.wmma_async_id, 0});
  }

  void service_mem_uop() {
    if (mem_uops_.empty()) {
      return;
    }
    auto uop = mem_uops_.front();
    mem_uops_.pop_front();
    auto& slot = operand_slots_.at(uop.slot_id);
    switch (uop.kind) {
    case MemUop::Kind::FillA:
      execute_fill(slot, uop.handle, TcuTarget::A, uop.separate_handle);
      mark_target_ready(slot, TcuTarget::A);
      break;
    case MemUop::Kind::FillB:
      execute_fill(slot, uop.handle, TcuTarget::B, uop.separate_handle);
      mark_target_ready(slot, TcuTarget::B);
      break;
    case MemUop::Kind::FillC:
      execute_fill(slot, uop.handle, TcuTarget::C, uop.separate_handle);
      mark_target_ready(slot, TcuTarget::C);
      break;
    case MemUop::Kind::StoreC: {
      std::vector<CMem::packet_t> packets;
      if (!slot.cmem.dump_tile(slot.fmt_c, &packets)) {
        std::abort();
      }
      for (uint32_t i = 0; i < packets.size(); ++i) {
        Core::TmemPacket packet;
        std::copy_n(packets.at(i).begin(), packets.at(i).size(), packet.bytes.begin());
        if (!core_->tmem_write_packet(uop.handle, i, packet)) {
          std::abort();
        }
      }
      slot.store_pending = false;
      slot.busy = false;
      core_->async_tensor_complete(uop.async_id);
      release_slot(uop.slot_id);
      return;
    }
    default:
      std::abort();
    }

    auto pending_it = pending_load_uops_.find(uop.async_id);
    if (pending_it != pending_load_uops_.end()) {
      if (pending_it->second == 0) {
        std::abort();
      }
      --pending_it->second;
      if (pending_it->second == 0) {
        pending_load_uops_.erase(pending_it);
        core_->async_tensor_complete(uop.async_id);
      }
    }
  }

  void dispatch_compute_uop() {
    if (!tensorcore_.ready(true)) {
      return;
    }
    if (!active_wmma_job_valid_) {
      if (pending_wmma_jobs_.empty()) {
        return;
      }
      active_wmma_job_ = pending_wmma_jobs_.front();
      pending_wmma_jobs_.pop_front();
      prime_wmma_issue_budget();
    }

    auto& job = active_wmma_job_;
    auto& slot = operand_slots_.at(job.slot_id);

    uint32_t storage_k = (job.next_uop < kSubtilesPerTile) ? 0 : 1;
    uint32_t c_subtile_id = job.next_uop & (kSubtilesPerTile - 1);
    uint32_t storage_m = c_subtile_id / 2;
    uint32_t storage_n = c_subtile_id % 2;
    uint16_t a_block[kPrimitiveDim][kPrimitiveDim] = {};
    uint16_t b_block[kPrimitiveDim][kPrimitiveDim] = {};
    uint32_t zero_c[kPrimitiveDim][kPrimitiveDim] = {};

    slot.amem.read_primitive(storage_m, storage_k, a_block);
    slot.bmem.read_primitive(storage_k, storage_n, b_block);

    TensorCoreMeta meta{};
    meta.wid = job.wid;
    meta.async_id = job.async_id;
    meta.slot_id = job.slot_id;
    meta.c_subtile_id = c_subtile_id;
    meta.valid = true;

    tensorcore_fmt_c_ = job.fmt_c;
    configure_open_tensorcore_precision(job.fmt_c);
    tensorcore_.push_uop(a_block, b_block, zero_c, meta);

    ++job.next_uop;
    if (job.next_uop >= kWmmaPrimitiveCount) {
      active_wmma_job_valid_ = false;
      active_wmma_job_budget_ = 0;
      return;
    }
    if (active_wmma_job_budget_ > 0) {
      --active_wmma_job_budget_;
    }
    if (active_wmma_job_budget_ == 0 && !pending_wmma_jobs_.empty()) {
      pending_wmma_jobs_.push_back(job);
      active_wmma_job_valid_ = false;
    }
  }

  void prime_wmma_issue_budget() {
    active_wmma_job_valid_ = true;
    auto remaining_uops = kWmmaPrimitiveCount - active_wmma_job_.next_uop;
    if (pending_wmma_jobs_.empty()) {
      active_wmma_job_budget_ = remaining_uops;
      return;
    }
    active_wmma_job_budget_ = std::min<uint32_t>(kWmmaIssueBurst, remaining_uops);
  }

  void tick_tensorcore() {
    configure_open_tensorcore_precision(tensorcore_fmt_c_);
    tensorcore_.tick(true);

    TensorCoreRetire retire;
    if (!tensorcore_.pop_retired(&retire)) {
      return;
    }
    retire_primitive(retire);
  }

  void retire_primitive(const TensorCoreRetire& retire) {
    if (!retire.valid || !retire.meta.valid) {
      return;
    }
    auto& slot = operand_slots_.at(retire.meta.slot_id);
    auto subtile_id = retire.meta.c_subtile_id;

    for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
      for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
        auto& accum = slot.accum_fp22.at(subtile_id).at(i).at(j);
        accum = add_fp22_raw(accum, retire.fp22_out[i][j]);
      }
    }

    ++slot.retired_wmma_uops;
    ++slot.retired_c_subtiles.at(subtile_id);
    if (slot.retired_c_subtiles.at(subtile_id) == 2) {
      flush_slot_subtile(slot, subtile_id);
    }
    if (slot.retired_wmma_uops >= kWmmaPrimitiveCount) {
      slot.wmma_pending = false;
      slot.busy = false;
      slot.cmem_final_valid = true;
      core_->async_tensor_complete(retire.meta.async_id);
    }
  }

  TensorUnit* simobject_;
  Core*       core_;
  Arch        arch_;
  PerfStats   perf_stats_;
  std::array<OperandSlot, kNumOperandSlots> operand_slots_;
  std::deque<MemUop> mem_uops_;
  std::unordered_map<uint32_t, uint32_t> pending_load_uops_;
  std::deque<PendingWmmaJob> pending_wmma_jobs_;
  PendingWmmaJob active_wmma_job_;
  bool active_wmma_job_valid_ = false;
  uint32_t active_wmma_job_budget_ = 0;
  TensorCoreTop tensorcore_;
  uint32_t tensorcore_fmt_c_ = vt::fp16::id;
};

op_string_t vortex::op_string(TcuType tcu_type, IntrTcuArgs args) {
  auto fmt_a = args.fmt_a ? args.fmt_a : args.fmt_ab;
  auto fmt_b = args.fmt_b ? args.fmt_b : args.fmt_ab;
  switch (tcu_type) {
  case TcuType::TMEM_ALLOC:
    return {"TMEM_ALLOC." + std::to_string(args.bank_span), ""};
  case TcuType::TMEM_FREE:
    return {"TMEM_FREE", ""};
  case TcuType::TMEM_REL_PERMIT:
    return {"TMEM_REL_PERMIT", ""};
  case TcuType::TMA_LOAD:
    return {"TMA_LOAD" + std::string(args.transpose_b ? ".TB" : ""), ""};
  case TcuType::TMA_STORE:
    return {"TMA_STORE", ""};
  case TcuType::TC_COMMIT:
    return {"TC_COMMIT", std::to_string(args.barrier_id)};
  case TcuType::TC_FENCE:
    return {std::string("TC_FENCE.") + (args.fence_mode == TcuFenceMode::After ? "AFTER" : "BEFORE"), ""};
  case TcuType::TC_WAIT:
    return {"TC_WAIT", ""};
  case TcuType::TMEM_SHIFT:
    return {"TMEM_SHIFT", ""};
  case TcuType::MMA_LOAD:
    if (args.macro_op) {
      return {"MMA_LOAD.ASYNC." + std::to_string(args.descriptor) + "." + std::to_string(static_cast<uint32_t>(args.target)), ""};
    }
    return {"MMA_LOAD." + std::string(vt::fmt_string(fmt_a)) + "." + std::string(vt::fmt_string(fmt_b)) + "." + std::string(vt::fmt_string(args.fmt_c))
          + "." + std::to_string(static_cast<uint32_t>(args.target)), ""};
  case TcuType::MMA_STORE:
    if (args.macro_op) {
      return {"MMA_STORE.ASYNC." + std::to_string(args.descriptor) + "." + std::to_string(static_cast<uint32_t>(args.target)), ""};
    }
    return {"MMA_STORE." + std::string(vt::fmt_string(args.fmt_c)), ""};
  case TcuType::MBAR_INIT:
    return {"MBAR_INIT", ""};
  case TcuType::MBAR_ARRIVE:
    return {"MBAR_ARRIVE", ""};
  case TcuType::MBAR_WAIT:
    return {"MBAR_WAIT", ""};
  case TcuType::TMA_WAIT:
    return {"TMA_WAIT", ""};
  case TcuType::WMMA:
    if (args.macro_op) {
      return {"WMMA.ASYNC." + std::to_string(args.descriptor), ""};
    }
    return {"WMMA." + std::string(vt::fmt_string(fmt_a)) + "." + std::string(vt::fmt_string(fmt_b)) + "." + std::string(vt::fmt_string(args.fmt_c))
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
                      IntrTcuArgs args,
                      const std::vector<reg_data_t>& rs1_data,
                      const std::vector<reg_data_t>& rs2_data,
                      const std::vector<reg_data_t>& rs3_data,
                      std::vector<reg_data_t>& rd_data,
                      ExeTraceData* trace_data) {
  impl_->wmma(wid, args, rs1_data, rs2_data, rs3_data, rd_data, trace_data);
}

uint32_t TensorUnit::scheduler_score(uint32_t wid, TcuType tcu_type, IntrTcuArgs args) const {
  return impl_->scheduler_score(wid, tcu_type, args);
}

const TensorUnit::PerfStats& TensorUnit::perf_stats() const {
  return impl_->perf_stats();
}
