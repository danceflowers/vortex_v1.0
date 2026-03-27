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
#include <limits>
#include <unordered_map>
#include <vector>
#include <rvfloats.h>

#include "core.h"
#include "tensor_cfg.h"
#include "tmem_window_planner.h"
#include "open_tensorcore/amem.h"
#include "open_tensorcore/bmem.h"
#include "open_tensorcore/cmem.h"
#include "open_tensorcore/meta_mem.h"
#include "open_tensorcore/sparse_select.h"
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

uint32_t a_packet_count(uint32_t fmt_a) {
  return (fmt_a == vt::fp16::id) ? 8 : 4;
}

uint32_t b_packet_count(uint32_t fmt_b, uint32_t sparse_mode) {
  uint32_t dense = (fmt_b == vt::fp16::id) ? 8 : 4;
  if (sparse_mode == vt::sparse_2_4) {
    return dense * 2;
  }
  if (sparse_mode == vt::sparse_1_4) {
    return dense * 4;
  }
  return dense;
}

uint32_t meta_packet_count(uint32_t sparse_mode) {
  return (sparse_mode == vt::sparse_none) ? 0 : MetaMem::packet_count();
}

uint32_t meta_shadow_window_id(uint32_t window_id) {
  return window_id | 0x80000000u;
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
  auto a_packets = a_packet_count(fmt_a);
  auto b_packets = b_packet_count(fmt_b, args.sparse_mode);
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
  static constexpr uint32_t kAmemWriteBeatsPerCycle = 1;
  static constexpr uint32_t kBmemWriteBeatsPerCycle = 1;
  static constexpr uint32_t kCmemWriteBeatsPerCycle = 1;
  static constexpr uint32_t kCmemReadBeatsPerCycle = 1;
  static constexpr uint32_t kMetaWriteBeatsPerCycle = 1;
  struct OperandSlot {
    uint32_t owner_wid = 0;
    uint32_t descriptor = 0xffffffffu;
    uint32_t fmt_a = 0;
    uint32_t fmt_b = 0;
    uint32_t fmt_c = 0;
    uint32_t sparse_mode = 0;
    uint32_t wmma_async_id = 0;
    uint32_t store_async_id = 0;
    uint32_t c_wmma_inflight = 0;
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
    bool c_dirty = false;
    bool store_pending = false;

    void reset() {
      owner_wid = 0;
      descriptor = 0xffffffffu;
      fmt_a = 0;
      fmt_b = 0;
      fmt_c = 0;
      sparse_mode = 0;
      wmma_async_id = 0;
      store_async_id = 0;
      c_wmma_inflight = 0;
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
      c_dirty = false;
      store_pending = false;
    }
  };

  struct PendingWmmaJob {
    uint32_t wid = 0;
    uint32_t ab_slot_id = 0;
    uint32_t c_slot_id = 0;
    uint32_t fmt_a = 0;
    uint32_t fmt_b = 0;
    uint32_t fmt_c = 0;
    uint32_t sparse_mode = 0;
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
    uint32_t window_id = 0;
    uint32_t tile_idx = 0;
    uint32_t packets_per_tile = 0;
    uint32_t async_id = 0;
    bool separate_handle = false;
    uint32_t remaining_tmem_reads = 0;
    uint32_t remaining_tmem_writes = 0;
    uint32_t remaining_amem_writes = 0;
    uint32_t remaining_bmem_writes = 0;
    uint32_t remaining_cmem_writes = 0;
    uint32_t remaining_cmem_reads = 0;
    uint32_t remaining_meta_writes = 0;
    uint32_t next_payload_packet_idx = 0;
    uint32_t next_meta_packet_idx = 0;
    std::vector<Core::TmemPacket> staged_payload_packets;
    std::vector<Core::TmemPacket> staged_meta_packets;
    uint32_t next_store_packet_idx = 0;
    uint32_t staged_store_packet_cursor = 0;
    std::vector<Core::TmemPacket> staged_store_packets;
    std::array<std::array<uint32_t, kPrimitiveDim>, kPrimitiveDim> staged_store_left_subtile = {};
    bool staged_store_left_valid = false;
  };

  enum class WindowCursorRole : uint8_t {
    LoadA = 0,
    LoadB,
    LoadC,
    StoreC,
  };

  struct WindowCursorKey {
    uint32_t handle = 0;
    uint32_t window_id = 0;
    WindowCursorRole role = WindowCursorRole::LoadA;

    bool operator==(const WindowCursorKey& other) const {
      return handle == other.handle
          && window_id == other.window_id
          && role == other.role;
    }
  };

  struct WindowCursorKeyHash {
    size_t operator()(const WindowCursorKey& key) const {
      size_t h1 = std::hash<uint32_t>{}(key.handle);
      size_t h2 = std::hash<uint32_t>{}(key.window_id);
      size_t h3 = std::hash<uint32_t>{}(static_cast<uint32_t>(key.role));
      return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
  };

  struct WindowCursorState {
    uint32_t epoch = 0;
    uint32_t next_tile_idx = 0;
    uint32_t next_packet_idx_in_tile = 0;

    void reset(uint32_t new_epoch) {
      epoch = new_epoch;
      next_tile_idx = 0;
      next_packet_idx_in_tile = 0;
    }
  };

  enum class NoWmmaReadyReason : uint8_t {
    JobBuilderEmpty = 0,
    WaitingForMmaLoad,
    WaitingForHandleAlloc,
    WaitingForSlotRelease,
  };

  enum class SlotReleaseReason : uint8_t {
    None = 0,
    CWmmaInflightDrain,
    CAccumLiveOnly,
    CDirtyFlushOnly,
    CStorePending,
    AbWmmaPendingClear,
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
    amem_.reset();
    bmem_.reset();
    cmem_.reset();
    metamem_.reset();
    for (auto& slot : operand_slots_) {
      slot.reset();
    }
    mem_ops_.clear();
    pending_mem_ops_.clear();
    pending_wmma_uops_.clear();
    pending_wmma_jobs_.clear();
    active_wmma_job_ = {};
    active_wmma_job_valid_ = false;
    window_cursors_.clear();
    tensorcore_.reset();
    tensorcore_fmt_c_ = vt::fp16::id;
    mem_port_cycle_ = std::numeric_limits<uint64_t>::max();
    amem_write_budget_ = 0;
    bmem_write_budget_ = 0;
    cmem_write_budget_ = 0;
    cmem_read_budget_ = 0;
    meta_write_budget_ = 0;
  }

  bool reserve_window_cursor(uint32_t handle,
                             uint32_t window_id,
                             WindowCursorRole role,
                             uint32_t* tile_idx,
                             uint32_t* packets_per_tile) {
    const TmemWindowPlan* window = nullptr;
    if (!core_->lookup_tmem_window(handle, window_id, &window)) {
      if (tile_idx) {
        *tile_idx = 0;
      }
      if (packets_per_tile) {
        *packets_per_tile = 0;
      }
      return false;
    }

    uint32_t epoch = 0;
    if (!core_->tmem_window_epoch(handle, &epoch)) {
      epoch = 0;
    }

    auto& state = window_cursors_[WindowCursorKey{handle, window_id, role}];
    if (state.epoch != epoch) {
      state.reset(epoch);
    }
    if (tile_idx) {
      *tile_idx = state.next_tile_idx;
    }
    if (packets_per_tile) {
      *packets_per_tile = window->packets_per_tile;
    }
    ++state.next_tile_idx;
    state.next_packet_idx_in_tile = 0;
    return true;
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

    service_mem_ops();
    dispatch_compute_uop();
    sample_pending_wmma_depth();
    if (tensorcore_.active()) {
      ++perf_stats_.tc_active_cycles;
    }
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
    int32_t slot_id = -1;
    if (args.target == TcuTarget::None) {
      slot_id = find_free_slot();
    } else {
      slot_id = find_free_slot();
    }
    if (slot_id < 0) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    auto& slot = operand_slots_.at(slot_id);
    if (!slot.valid) {
      init_slot_for_descriptor(slot_id, slot, wid, args);
    } else {
      ensure_slot_matches(slot, wid, args);
    }
    if (args.target != TcuTarget::None && !slot_target_available(slot, args.target)) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    execute_fill(static_cast<uint32_t>(slot_id), slot, handle, args.window_id, args.target, false);
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
    if (!cmem_.dump_tile(static_cast<uint32_t>(slot_id), slot.fmt_c, &packets)) {
      std::abort();
    }
    IntrTcuArgs resolved_args{};
    resolved_args.fmt_a = slot.fmt_a;
    resolved_args.fmt_b = slot.fmt_b;
    resolved_args.fmt_ab = (slot.fmt_a == slot.fmt_b) ? slot.fmt_a : 0;
    resolved_args.fmt_c = slot.fmt_c;
    resolved_args.sparse_mode = slot.sparse_mode;
    resolved_args.target = TcuTarget::C;
    auto packet_offset = (args.target == TcuTarget::None) ? target_packet_offset(resolved_args) : 0;
    const TmemWindowPlan* window = nullptr;
    bool use_window = (args.target != TcuTarget::None)
                   && core_->lookup_tmem_window(handle, args.window_id, &window)
                   && window->fmt != 0
                   && window->packets_per_tile == packets.size();
    for (uint32_t i = 0; i < packets.size(); ++i) {
      Core::TmemPacket packet;
      std::copy_n(packets.at(i).begin(), packets.at(i).size(), packet.bytes.begin());
      if (use_window) {
        if (!core_->tmem_write_window_packet(handle, args.window_id, i, packet)) {
          std::abort();
        }
      } else if (!core_->tmem_write_packet(handle, packet_offset + i, packet)) {
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
      enqueue_async_wmma(wid, args, fmt_a, fmt_b, args.fmt_c, trace_data);
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
    if (args.slot_id >= operand_slots_.size()) {
      return 0;
    }

    switch (tcu_type) {
    case TcuType::WMMA: {
      auto fmt_a = args.fmt_a ? args.fmt_a : args.fmt_ab;
      auto fmt_b = args.fmt_b ? args.fmt_b : args.fmt_ab;
      if (!use_open_tensorcore(fmt_a, fmt_b, args.fmt_c)) {
        return 0;
      }
      if (args.c_slot_id >= operand_slots_.size()) {
        return 0;
      }
      const auto& ab_slot = operand_slots_.at(args.slot_id);
      const auto& c_slot = operand_slots_.at(args.c_slot_id);
      if (!ab_slot.valid || ab_slot.owner_wid != wid || ab_slot.descriptor != args.descriptor) {
        return 0;
      }
      if (!c_slot.valid || c_slot.owner_wid != wid || c_slot.descriptor != args.descriptor) {
        return 0;
      }
      return (ab_slot.a_ready && ab_slot.b_ready && !ab_slot.wmma_pending
           && c_slot.c_ready && !c_slot.store_pending) ? 5 : 0;
    }
    case TcuType::MMA_LOAD:
      return (core_->tmem_handle_ready_for_mma_load(args.runtime_handle, args.target, args.sparse_mode)
           && can_issue_mma_load(wid, args)) ? 4 : 0;
    case TcuType::MMA_STORE: {
      if (!core_->tmem_handle_ready_for_mma_store(args.runtime_handle)) {
        return 0;
      }
      const auto& slot = operand_slots_.at(args.slot_id);
      if (!slot.valid || slot.owner_wid != wid || slot.descriptor != args.descriptor) {
        return 0;
      }
      bool ready = !slot.store_pending && slot.cmem_final_valid;
      return ready ? 3 : 0;
    }
    default:
      return 1;
    }
  }

  TensorUnit::IssueBlockReason classify_issue_block(uint32_t wid, TcuType tcu_type, IntrTcuArgs args) const {
    if (!args.macro_op) {
      return TensorUnit::IssueBlockReason::None;
    }
    if (args.slot_id >= operand_slots_.size()) {
      return TensorUnit::IssueBlockReason::SlotBusy;
    }

    switch (tcu_type) {
    case TcuType::WMMA: {
      if (args.c_slot_id >= operand_slots_.size()) {
        return TensorUnit::IssueBlockReason::SlotBusy;
      }
      const auto& ab_slot = operand_slots_.at(args.slot_id);
      const auto& c_slot = operand_slots_.at(args.c_slot_id);
      if (!ab_slot.valid || ab_slot.owner_wid != wid || ab_slot.descriptor != args.descriptor
       || !c_slot.valid || c_slot.owner_wid != wid || c_slot.descriptor != args.descriptor) {
        return TensorUnit::IssueBlockReason::SlotBusy;
      }
      if (!ab_slot.a_ready) {
        return TensorUnit::IssueBlockReason::ANotReady;
      }
      if (!ab_slot.b_ready) {
        return TensorUnit::IssueBlockReason::BNotReady;
      }
      if (!c_slot.c_ready) {
        return TensorUnit::IssueBlockReason::CNotReady;
      }
      if (ab_slot.wmma_pending || c_slot.store_pending) {
        return TensorUnit::IssueBlockReason::SlotBusy;
      }
      return TensorUnit::IssueBlockReason::None;
    }
    case TcuType::MMA_LOAD: {
      auto handle_reason = core_->tmem_handle_load_block_reason(args.runtime_handle, args.target, args.sparse_mode);
      switch (handle_reason) {
      case Core::TmemHandleBlockReason::MetaNotReady:
        return TensorUnit::IssueBlockReason::AMetaNotReady;
      case Core::TmemHandleBlockReason::BusyTmaLoad:
        return TensorUnit::IssueBlockReason::HandleBusyDueToTmaLoad;
      case Core::TmemHandleBlockReason::BusyTmaStoreOrShift:
        return TensorUnit::IssueBlockReason::HandleBusyDueToTmaStoreOrShift;
      case Core::TmemHandleBlockReason::PayloadNotReady:
        return TensorUnit::IssueBlockReason::MmaLoadHandleNotReady;
      case Core::TmemHandleBlockReason::Invalid:
        return TensorUnit::IssueBlockReason::HandleReuse;
      case Core::TmemHandleBlockReason::None:
        break;
      }
      if (!can_issue_mma_load(wid, args)) {
        return TensorUnit::IssueBlockReason::SlotBusy;
      }
      return TensorUnit::IssueBlockReason::None;
    }
    case TcuType::MMA_STORE: {
      auto handle_reason = core_->tmem_handle_store_block_reason(args.runtime_handle);
      if (handle_reason == Core::TmemHandleBlockReason::BusyTmaLoad) {
        return TensorUnit::IssueBlockReason::HandleBusyDueToTmaLoad;
      }
      if (handle_reason == Core::TmemHandleBlockReason::BusyTmaStoreOrShift) {
        return TensorUnit::IssueBlockReason::HandleBusyDueToTmaStoreOrShift;
      }
      if (handle_reason == Core::TmemHandleBlockReason::Invalid) {
        return TensorUnit::IssueBlockReason::HandleReuse;
      }
      const auto& slot = operand_slots_.at(args.slot_id);
      bool ready = slot.valid && slot.owner_wid == wid && slot.descriptor == args.descriptor
                && !slot.store_pending
                && slot.cmem_final_valid;
      return ready ? TensorUnit::IssueBlockReason::None
                   : TensorUnit::IssueBlockReason::SlotBusy;
    }
    default:
      return TensorUnit::IssueBlockReason::None;
    }
  }

  void record_issue_stall(TensorUnit::IssueBlockReason reason) {
    switch (reason) {
    case TensorUnit::IssueBlockReason::ANotReady:
      ++perf_stats_.stall_a_not_ready;
      break;
    case TensorUnit::IssueBlockReason::BNotReady:
      ++perf_stats_.stall_b_not_ready;
      break;
    case TensorUnit::IssueBlockReason::CNotReady:
      ++perf_stats_.stall_c_not_ready;
      break;
    case TensorUnit::IssueBlockReason::TcBusy:
      ++perf_stats_.stall_tc_busy;
      break;
    case TensorUnit::IssueBlockReason::NoTensorInstrCandidate:
      ++perf_stats_.stall_no_tensor_instr_candidate;
      break;
    case TensorUnit::IssueBlockReason::MmaLoadHandleNotReady:
      ++perf_stats_.stall_mma_load_handle_not_ready;
      break;
    case TensorUnit::IssueBlockReason::HandleBusyDueToTmaLoad:
      ++perf_stats_.stall_handle_reuse;
      ++perf_stats_.stall_handle_busy_due_to_tma_load;
      break;
    case TensorUnit::IssueBlockReason::HandleBusyDueToTmaStoreOrShift:
      ++perf_stats_.stall_handle_reuse;
      ++perf_stats_.stall_handle_busy_due_to_tma_store_or_shift;
      break;
    case TensorUnit::IssueBlockReason::HandleReuse:
      ++perf_stats_.stall_handle_reuse;
      break;
    case TensorUnit::IssueBlockReason::SlotBusy:
      ++perf_stats_.stall_slot_busy;
      break;
    case TensorUnit::IssueBlockReason::AMetaNotReady:
      ++perf_stats_.stall_a_meta_not_ready;
      ++perf_stats_.stall_mma_load_handle_not_ready;
      break;
    case TensorUnit::IssueBlockReason::None:
    default:
      break;
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

  void release_slot(uint32_t slot_id) {
    clear_slot_storage(slot_id);
    operand_slots_.at(slot_id).reset();
  }

  static bool slot_can_rebind(const OperandSlot& slot) {
    return !slot.valid || (!slot_has_pending_work(slot) && !slot.c_dirty);
  }

  static bool slot_has_pending_work(const OperandSlot& slot) {
    return ab_side_has_pending(slot) || c_side_has_pending(slot);
  }

  static bool ab_side_has_pending(const OperandSlot& slot) {
    return slot.a_pending || slot.b_pending || slot.wmma_pending;
  }

  static bool c_side_has_pending(const OperandSlot& slot) {
    return slot.c_pending || slot.store_pending || slot.c_wmma_inflight != 0;
  }

  static bool ab_side_target_available(const OperandSlot& slot, TcuTarget target) {
    switch (target) {
    case TcuTarget::A:
      return !slot.a_pending && !slot.wmma_pending;
    case TcuTarget::B:
      return !slot.b_pending && !slot.wmma_pending;
    default:
      return false;
    }
  }

  static bool c_side_target_available(const OperandSlot& slot) {
    return !slot.c_pending && !slot.store_pending
        && !slot.c_dirty && slot.c_wmma_inflight == 0;
  }

  static bool slot_target_available(const OperandSlot& slot, TcuTarget target) {
    switch (target) {
    case TcuTarget::A:
    case TcuTarget::B:
      return ab_side_target_available(slot, target);
    case TcuTarget::C:
      return c_side_target_available(slot);
    default:
      return false;
    }
  }

  void reset_mem_port_budgets() {
    mem_port_cycle_ = core_->current_cycle();
    amem_write_budget_ = kAmemWriteBeatsPerCycle;
    bmem_write_budget_ = kBmemWriteBeatsPerCycle;
    cmem_write_budget_ = kCmemWriteBeatsPerCycle;
    cmem_read_budget_ = kCmemReadBeatsPerCycle;
    meta_write_budget_ = kMetaWriteBeatsPerCycle;
  }

  void ensure_mem_port_budgets() {
    if (mem_port_cycle_ != core_->current_cycle()) {
      reset_mem_port_budgets();
    }
  }

  static bool mem_op_complete(const MemUop& op) {
    return op.remaining_tmem_reads == 0
        && op.remaining_tmem_writes == 0
        && op.remaining_amem_writes == 0
        && op.remaining_bmem_writes == 0
        && op.remaining_cmem_writes == 0
        && op.remaining_cmem_reads == 0
        && op.remaining_meta_writes == 0
        && op.staged_store_packets.empty()
        && !op.staged_store_left_valid;
  }

  uint32_t current_pending_wmma_depth() const {
    return pending_wmma_jobs_.size() + (active_wmma_job_valid_ ? 1u : 0u);
  }

  void sample_pending_wmma_depth() {
    auto depth = current_pending_wmma_depth();
    perf_stats_.pending_wmma_jobs_max = std::max<uint64_t>(perf_stats_.pending_wmma_jobs_max, depth);
    switch (depth) {
    case 0:
      ++perf_stats_.pending_wmma_depth_cycles_0;
      break;
    case 1:
      ++perf_stats_.pending_wmma_depth_cycles_1;
      break;
    case 2:
      ++perf_stats_.pending_wmma_depth_cycles_2;
      break;
    default:
      ++perf_stats_.pending_wmma_depth_cycles_3plus;
      break;
    }
  }

  bool has_inflight_mma_load_build() const {
    for (const auto& op : mem_ops_) {
      switch (op.kind) {
      case MemUop::Kind::FillA:
      case MemUop::Kind::FillB:
      case MemUop::Kind::FillC:
        return true;
      default:
        break;
      }
    }
    for (const auto& slot : operand_slots_) {
      if (slot.a_pending || slot.b_pending || slot.c_pending) {
        return true;
      }
    }
    return false;
  }

  SlotReleaseReason classify_slot_release_reason() const {
    for (const auto& slot : operand_slots_) {
      if (!slot.valid) {
        continue;
      }
      if (slot.wmma_pending) {
        return SlotReleaseReason::AbWmmaPendingClear;
      }
      if (slot.c_wmma_inflight != 0) {
        return SlotReleaseReason::CWmmaInflightDrain;
      }
      if (slot.store_pending) {
        return SlotReleaseReason::CStorePending;
      }
      if (slot.c_dirty) {
        return SlotReleaseReason::CDirtyFlushOnly;
      }
    }
    return SlotReleaseReason::None;
  }

  NoWmmaReadyReason classify_no_wmma_ready_reason() const {
    if (has_inflight_mma_load_build()) {
      return NoWmmaReadyReason::WaitingForMmaLoad;
    }
    if (core_->has_inflight_tma_handle_activity()) {
      return NoWmmaReadyReason::WaitingForHandleAlloc;
    }
    if (classify_slot_release_reason() != SlotReleaseReason::None) {
      return NoWmmaReadyReason::WaitingForSlotRelease;
    }
    return NoWmmaReadyReason::JobBuilderEmpty;
  }

  void record_no_wmma_ready_stall() {
    ++perf_stats_.stall_no_wmma_job_ready;
    switch (classify_no_wmma_ready_reason()) {
    case NoWmmaReadyReason::JobBuilderEmpty:
      ++perf_stats_.stall_no_wmma_job_builder_empty;
      break;
    case NoWmmaReadyReason::WaitingForMmaLoad:
      ++perf_stats_.stall_no_wmma_waiting_for_mma_load;
      break;
    case NoWmmaReadyReason::WaitingForHandleAlloc:
      ++perf_stats_.stall_no_wmma_waiting_for_handle_alloc;
      break;
    case NoWmmaReadyReason::WaitingForSlotRelease:
      ++perf_stats_.stall_no_wmma_waiting_for_slot_release;
      switch (classify_slot_release_reason()) {
      case SlotReleaseReason::CWmmaInflightDrain:
        ++perf_stats_.stall_no_wmma_waiting_for_c_wmma_inflight_drain;
        break;
      case SlotReleaseReason::CAccumLiveOnly:
        ++perf_stats_.stall_no_wmma_waiting_for_accum_live_only;
        break;
      case SlotReleaseReason::CDirtyFlushOnly:
        ++perf_stats_.stall_no_wmma_waiting_for_dirty_flush_only;
        break;
      case SlotReleaseReason::CStorePending:
        ++perf_stats_.stall_no_wmma_waiting_for_store_pending;
        break;
      case SlotReleaseReason::AbWmmaPendingClear:
        ++perf_stats_.stall_no_wmma_waiting_for_ab_wmma_pending_clear;
        break;
      case SlotReleaseReason::None:
        break;
      }
      break;
    }
  }

  bool can_issue_mma_load(uint32_t wid, const IntrTcuArgs& args) const {
    if (args.slot_id >= operand_slots_.size()) {
      return false;
    }
    const auto& slot = operand_slots_.at(args.slot_id);
    if (!slot.valid) {
      return true;
    }
    if (slot.owner_wid != wid || slot.descriptor != args.descriptor) {
      return slot_can_rebind(slot);
    }
    if (args.target == TcuTarget::None) {
      return !slot_has_pending_work(slot) && !slot.c_dirty;
    }
    return slot_target_available(slot, args.target);
  }

  void clear_slot_storage(uint32_t slot_id) {
    amem_.clear_slot(slot_id);
    bmem_.clear_slot(slot_id);
    cmem_.clear_slot(slot_id);
    metamem_.clear_slot(slot_id);
  }

  void init_slot_for_descriptor(uint32_t slot_id, OperandSlot& slot, uint32_t wid, const IntrTcuArgs& args) {
    clear_slot_storage(slot_id);
    slot.reset();
    slot.valid = true;
    slot.owner_wid = wid;
    slot.descriptor = args.descriptor;
    slot.fmt_a = args.fmt_a ? args.fmt_a : args.fmt_ab;
    slot.fmt_b = args.fmt_b ? args.fmt_b : args.fmt_ab;
    slot.fmt_c = args.fmt_c;
    slot.sparse_mode = args.sparse_mode;
  }

  void ensure_slot_matches(OperandSlot& slot, uint32_t wid, const IntrTcuArgs& args) {
    if (!slot.valid) {
      std::abort();
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
      slot.c_dirty = false;
      slot.c_wmma_inflight = 0;
      break;
    default:
      std::abort();
    }
    slot.busy = slot_has_pending_work(slot);
  }

  static TcuTarget mem_uop_target(MemUop::Kind kind) {
    switch (kind) {
    case MemUop::Kind::FillA:
      return TcuTarget::A;
    case MemUop::Kind::FillB:
      return TcuTarget::B;
    case MemUop::Kind::FillC:
    case MemUop::Kind::StoreC:
      return TcuTarget::C;
    default:
      std::abort();
    }
  }

  uint32_t fill_packet_offset(const OperandSlot& slot, const MemUop& uop) const {
    if (uop.separate_handle) {
      return 0;
    }
    IntrTcuArgs args{};
    args.fmt_a = slot.fmt_a;
    args.fmt_b = slot.fmt_b;
    args.fmt_c = slot.fmt_c;
    args.sparse_mode = slot.sparse_mode;
    args.target = mem_uop_target(uop.kind);
    return target_packet_offset(args);
  }

  uint32_t fill_payload_packet_count(const OperandSlot& slot, const MemUop& uop) const {
    switch (uop.kind) {
    case MemUop::Kind::FillA:
      return a_packet_count(slot.fmt_a);
    case MemUop::Kind::FillB:
      return b_packet_count(slot.fmt_b, slot.sparse_mode);
    case MemUop::Kind::FillC:
      return c_packet_count(slot.fmt_c);
    default:
      std::abort();
    }
  }

  uint32_t store_packet_offset(const OperandSlot& slot, const MemUop& uop) const {
    if (uop.separate_handle) {
      return 0;
    }
    IntrTcuArgs args{};
    args.fmt_a = slot.fmt_a;
    args.fmt_b = slot.fmt_b;
    args.fmt_c = slot.fmt_c;
    args.sparse_mode = slot.sparse_mode;
    args.target = TcuTarget::C;
    return target_packet_offset(args);
  }

  bool lookup_window_packet_idx(const MemUop& uop,
                                uint32_t local_packet_idx,
                                uint32_t* packet_idx) const {
    if (!uop.separate_handle || nullptr == packet_idx) {
      return false;
    }
    const TmemWindowPlan* window = nullptr;
    if (!core_->lookup_tmem_window(uop.handle, uop.window_id, &window)) {
      return false;
    }
    if (window->fmt == 0 || window->packets_per_tile == 0) {
      return false;
    }
    *packet_idx = (uop.tile_idx * window->packets_per_tile) + local_packet_idx;
    return true;
  }

  bool lookup_meta_window_packet_idx(const MemUop& uop,
                                     uint32_t local_packet_idx,
                                     uint32_t* packet_idx) const {
    if (!uop.separate_handle || nullptr == packet_idx) {
      return false;
    }
    const TmemWindowPlan* window = nullptr;
    if (!core_->lookup_tmem_window(uop.handle, meta_shadow_window_id(uop.window_id), &window)) {
      return false;
    }
    if (window->fmt == 0 || window->packets_per_tile == 0) {
      return false;
    }
    *packet_idx = (uop.tile_idx * window->packets_per_tile) + local_packet_idx;
    return true;
  }

  enum class LocalFillAction : uint8_t {
    None = 0,
    AData,
    BData,
    CData,
    Meta,
  };

  LocalFillAction select_fill_local_write(const MemUop& uop, const OperandSlot& slot) const {
    switch (uop.kind) {
    case MemUop::Kind::FillA:
      if (uop.remaining_amem_writes != 0) {
        auto beat_idx = AMem::fill_beats() - uop.remaining_amem_writes;
        auto packets_needed = AMem::packets_per_fill_beat(slot.fmt_a);
        if (uop.staged_payload_packets.size() >= (beat_idx + 1) * packets_needed) {
          return LocalFillAction::AData;
        }
      }
      if (uop.remaining_meta_writes != 0
       && uop.staged_meta_packets.size() >= MetaMem::packet_count()) {
        return LocalFillAction::Meta;
      }
      return LocalFillAction::None;
    case MemUop::Kind::FillB:
      if (uop.remaining_bmem_writes != 0) {
        auto beat_idx = BMem::fill_beats(slot.sparse_mode) - uop.remaining_bmem_writes;
        auto packets_needed = BMem::packets_per_fill_beat(slot.fmt_b, slot.sparse_mode);
        if (uop.staged_payload_packets.size() >= (beat_idx + 1) * packets_needed) {
          return LocalFillAction::BData;
        }
      }
      return LocalFillAction::None;
    case MemUop::Kind::FillC:
      if (uop.remaining_cmem_writes != 0) {
        auto beat_idx = CMem::fill_beats(slot.fmt_c) - uop.remaining_cmem_writes;
        auto packets_needed = CMem::packets_per_fill_group(slot.fmt_c);
        auto group_base = (beat_idx / 2) * packets_needed;
        if (uop.staged_payload_packets.size() >= group_base + packets_needed) {
          return LocalFillAction::CData;
        }
      }
      return LocalFillAction::None;
    default:
      return LocalFillAction::None;
    }
  }

  static AMem::packet_t to_amem_packet(const Core::TmemPacket& packet) {
    AMem::packet_t out{};
    std::copy_n(packet.bytes.begin(), out.size(), out.begin());
    return out;
  }

  static BMem::packet_t to_bmem_packet(const Core::TmemPacket& packet) {
    BMem::packet_t out{};
    std::copy_n(packet.bytes.begin(), out.size(), out.begin());
    return out;
  }

  static CMem::packet_t to_cmem_packet(const Core::TmemPacket& packet) {
    CMem::packet_t out{};
    std::copy_n(packet.bytes.begin(), out.size(), out.begin());
    return out;
  }

  static MetaMem::packet_t to_meta_packet(const Core::TmemPacket& packet) {
    MetaMem::packet_t out{};
    std::copy_n(packet.bytes.begin(), out.size(), out.begin());
    return out;
  }

  static Core::TmemPacket to_tmem_packet(const CMem::packet_t& packet) {
    Core::TmemPacket out;
    std::copy_n(packet.begin(), packet.size(), out.bytes.begin());
    return out;
  }

  bool perform_fill_local_write(MemUop& uop, const OperandSlot& slot, LocalFillAction action) {
    switch (action) {
    case LocalFillAction::AData: {
      auto beat_idx = AMem::fill_beats() - uop.remaining_amem_writes;
      auto packets_per_beat = AMem::packets_per_fill_beat(slot.fmt_a);
      std::vector<AMem::packet_t> packets;
      auto base = beat_idx * packets_per_beat;
      for (uint32_t i = 0; i < packets_per_beat; ++i) {
        packets.push_back(to_amem_packet(uop.staged_payload_packets.at(base + i)));
      }
      if (!amem_.write_fill_beat(uop.slot_id, slot.fmt_a, beat_idx, packets)) {
        std::abort();
      }
      --uop.remaining_amem_writes;
      return true;
    }
    case LocalFillAction::BData: {
      auto beat_idx = BMem::fill_beats(slot.sparse_mode) - uop.remaining_bmem_writes;
      auto packets_per_beat = BMem::packets_per_fill_beat(slot.fmt_b, slot.sparse_mode);
      std::vector<BMem::packet_t> packets;
      auto base = beat_idx * packets_per_beat;
      for (uint32_t i = 0; i < packets_per_beat; ++i) {
        packets.push_back(to_bmem_packet(uop.staged_payload_packets.at(base + i)));
      }
      if (!bmem_.write_fill_beat(uop.slot_id, slot.fmt_b, beat_idx, packets, slot.sparse_mode)) {
        std::abort();
      }
      --uop.remaining_bmem_writes;
      return true;
    }
    case LocalFillAction::CData: {
      auto beat_idx = CMem::fill_beats(slot.fmt_c) - uop.remaining_cmem_writes;
      auto packets_per_group = CMem::packets_per_fill_group(slot.fmt_c);
      auto group_base = (beat_idx / 2) * packets_per_group;
      std::vector<CMem::packet_t> packets;
      for (uint32_t i = 0; i < packets_per_group; ++i) {
        packets.push_back(to_cmem_packet(uop.staged_payload_packets.at(group_base + i)));
      }
      if (!cmem_.write_fill_beat(uop.slot_id, slot.fmt_c, beat_idx, packets)) {
        std::abort();
      }
      --uop.remaining_cmem_writes;
      return true;
    }
    case LocalFillAction::Meta: {
      if (!metamem_.write_fill_beat(uop.slot_id, to_meta_packet(uop.staged_meta_packets.front()))) {
        std::abort();
      }
      --uop.remaining_meta_writes;
      return true;
    }
    case LocalFillAction::None:
    default:
      return false;
    }
  }

  bool stage_fill_read(MemUop& uop, const OperandSlot& slot) {
    if (uop.remaining_tmem_reads == 0) {
      return false;
    }

    auto payload_packets = fill_payload_packet_count(slot, uop);
    Core::TmemPacket packet;
    if (uop.next_payload_packet_idx < payload_packets) {
      uint32_t packet_idx = 0;
      if (!lookup_window_packet_idx(uop, uop.next_payload_packet_idx, &packet_idx)) {
        packet_idx = fill_packet_offset(slot, uop) + uop.next_payload_packet_idx;
        if (!core_->tmem_read_packet(uop.handle, packet_idx, &packet)) {
          std::abort();
        }
      } else if (!core_->tmem_read_window_packet(uop.handle, uop.window_id, packet_idx, &packet)) {
        std::abort();
      }
      uop.staged_payload_packets.push_back(packet);
      ++uop.next_payload_packet_idx;
    } else {
      uint32_t meta_packet_idx = 0;
      if (!lookup_meta_window_packet_idx(uop, uop.next_meta_packet_idx, &meta_packet_idx)) {
        if (!core_->tmem_read_meta_packet(uop.handle, uop.next_meta_packet_idx, &packet)) {
          std::abort();
        }
      } else if (!core_->tmem_read_window_packet(uop.handle, meta_shadow_window_id(uop.window_id), meta_packet_idx, &packet)) {
        std::abort();
      }
      uop.staged_meta_packets.push_back(packet);
      ++uop.next_meta_packet_idx;
    }

    --uop.remaining_tmem_reads;
    return true;
  }

  bool try_acquire_fill_read_port(const MemUop& uop, const OperandSlot& slot) {
    auto payload_packets = fill_payload_packet_count(slot, uop);
    if (uop.next_payload_packet_idx < payload_packets) {
      uint32_t packet_idx = 0;
      if (!lookup_window_packet_idx(uop, uop.next_payload_packet_idx, &packet_idx)) {
        packet_idx = fill_packet_offset(slot, uop) + uop.next_payload_packet_idx;
        return core_->try_acquire_tmem_read_port(uop.handle, packet_idx);
      }
      return core_->try_acquire_tmem_window_read_port(uop.handle, uop.window_id, packet_idx);
    }
    uint32_t meta_packet_idx = 0;
    if (!lookup_meta_window_packet_idx(uop, uop.next_meta_packet_idx, &meta_packet_idx)) {
      return core_->try_acquire_tmem_read_meta_port(uop.handle, uop.next_meta_packet_idx);
    }
    return core_->try_acquire_tmem_window_read_port(uop.handle, meta_shadow_window_id(uop.window_id), meta_packet_idx);
  }

  void append_store_packets_for_subtile_pair(
      MemUop& uop,
      uint32_t fmt_c,
      const std::array<std::array<uint32_t, kPrimitiveDim>, kPrimitiveDim>& left,
      const uint32_t right[kPrimitiveDim][kPrimitiveDim]) {
    if (fmt_c == vt::fp32::id) {
      for (uint32_t row = 0; row < kPrimitiveDim; ++row) {
        CMem::packet_t packet{};
        for (uint32_t col = 0; col < CMem::kDim; ++col) {
          auto raw = (col < kPrimitiveDim) ? left[row][col] : right[row][col - kPrimitiveDim];
          auto bits = fp22_to_fp32(raw);
          auto off = col * 4;
          packet.at(off + 0) = bits & 0xff;
          packet.at(off + 1) = (bits >> 8) & 0xff;
          packet.at(off + 2) = (bits >> 16) & 0xff;
          packet.at(off + 3) = (bits >> 24) & 0xff;
        }
        uop.staged_store_packets.push_back(to_tmem_packet(packet));
      }
      return;
    }

    for (uint32_t packet_idx = 0; packet_idx < 4; ++packet_idx) {
      CMem::packet_t packet{};
      for (uint32_t elem = 0; elem < 32; ++elem) {
        auto row = packet_idx * 2 + (elem / 16);
        auto col = elem % 16;
        auto raw = (col < kPrimitiveDim) ? left[row][col] : right[row][col - kPrimitiveDim];
        auto bits = fp22_to_fp16(raw);
        auto off = elem * 2;
        packet.at(off + 0) = bits & 0xff;
        packet.at(off + 1) = (bits >> 8) & 0xff;
      }
      uop.staged_store_packets.push_back(to_tmem_packet(packet));
    }
  }

  bool stage_store_read(MemUop& uop, const OperandSlot& slot) {
    if (uop.remaining_cmem_reads == 0) {
      return false;
    }

    auto beat_idx = CMem::dump_beats(slot.fmt_c) - uop.remaining_cmem_reads;
    uint32_t subtile[kPrimitiveDim][kPrimitiveDim] = {};
    cmem_.read_subtile_fp22(uop.slot_id, beat_idx, subtile);

    if ((beat_idx & 1u) == 0) {
      for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
        for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
          uop.staged_store_left_subtile[i][j] = subtile[i][j];
        }
      }
      uop.staged_store_left_valid = true;
    } else {
      if (!uop.staged_store_left_valid) {
        std::abort();
      }
      append_store_packets_for_subtile_pair(uop, slot.fmt_c, uop.staged_store_left_subtile, subtile);
      uop.staged_store_left_valid = false;
    }

    --uop.remaining_cmem_reads;
    return true;
  }

  bool emit_store_packet(MemUop& uop, const OperandSlot& slot) {
    if (uop.staged_store_packet_cursor >= uop.staged_store_packets.size()
     || uop.remaining_tmem_writes == 0) {
      return false;
    }

    uint32_t packet_idx = 0;
    if (!lookup_window_packet_idx(uop, uop.next_store_packet_idx, &packet_idx)) {
      packet_idx = store_packet_offset(slot, uop) + uop.next_store_packet_idx;
      const auto& packet = uop.staged_store_packets.at(uop.staged_store_packet_cursor);
      if (!core_->tmem_write_packet(uop.handle, packet_idx, packet)) {
        std::abort();
      }
    } else {
      const auto& packet = uop.staged_store_packets.at(uop.staged_store_packet_cursor);
      if (!core_->tmem_write_window_packet(uop.handle, uop.window_id, packet_idx, packet)) {
        std::abort();
      }
    }
    ++uop.next_store_packet_idx;
    ++uop.staged_store_packet_cursor;
    --uop.remaining_tmem_writes;
    if (uop.staged_store_packet_cursor >= uop.staged_store_packets.size()) {
      uop.staged_store_packets.clear();
      uop.staged_store_packet_cursor = 0;
    }
    return true;
  }

  bool try_acquire_store_write_port(const MemUop& uop, const OperandSlot& slot) {
    uint32_t packet_idx = 0;
    if (!lookup_window_packet_idx(uop, uop.next_store_packet_idx, &packet_idx)) {
      packet_idx = store_packet_offset(slot, uop) + uop.next_store_packet_idx;
      return core_->try_acquire_tmem_write_port(uop.handle, packet_idx);
    }
    return core_->try_acquire_tmem_window_write_port(uop.handle, uop.window_id, packet_idx);
  }

  void execute_fill(uint32_t slot_id,
                    const OperandSlot& slot,
                    uint32_t handle,
                    uint32_t window_id,
                    TcuTarget target,
                    bool separate_handle) {
    IntrTcuArgs resolved_args{};
    resolved_args.descriptor = slot.descriptor;
    resolved_args.fmt_a = slot.fmt_a;
    resolved_args.fmt_b = slot.fmt_b;
    resolved_args.fmt_ab = (slot.fmt_a == slot.fmt_b) ? slot.fmt_a : 0;
    resolved_args.fmt_c = slot.fmt_c;
    resolved_args.sparse_mode = slot.sparse_mode;
    resolved_args.target = target;

    auto packets_needed = (target == TcuTarget::C)
                        ? c_packet_count(slot.fmt_c)
                        : ((target == TcuTarget::A)
                           ? a_packet_count(slot.fmt_a)
                           : b_packet_count(slot.fmt_b, slot.sparse_mode));
    auto packet_offset = separate_handle ? 0 : target_packet_offset(resolved_args);
    std::vector<Core::TmemPacket> packets(packets_needed);
    const TmemWindowPlan* window = nullptr;
    bool use_window = separate_handle
                   && core_->lookup_tmem_window(handle, window_id, &window)
                   && window->fmt != 0
                   && window->packets_per_tile == packets_needed;
    for (uint32_t i = 0; i < packets_needed; ++i) {
      if (use_window) {
        if (!core_->tmem_read_window_packet(handle, window_id, i, &packets.at(i))) {
          std::abort();
        }
      } else if (!core_->tmem_read_packet(handle, packet_offset + i, &packets.at(i))) {
        std::abort();
      }
    }

    switch (target) {
    case TcuTarget::A: {
      auto packet_bytes = copy_packets<AMem::packet_t>(packets);
      if (!amem_.fill_tile(slot_id, slot.fmt_a, packet_bytes)) {
        std::abort();
      }
      if (slot.sparse_mode != vt::sparse_none) {
        auto meta_packets_needed = meta_packet_count(slot.sparse_mode);
        std::vector<Core::TmemPacket> meta_packets(meta_packets_needed);
        for (uint32_t i = 0; i < meta_packets_needed; ++i) {
          uint32_t meta_packet_idx = i;
          const TmemWindowPlan* meta_window = nullptr;
          bool use_meta_window = separate_handle
                              && core_->lookup_tmem_window(handle, meta_shadow_window_id(window_id), &meta_window)
                              && meta_window->fmt != 0
                              && meta_window->packets_per_tile == meta_packets_needed;
          if (use_meta_window) {
            if (!core_->tmem_read_window_packet(handle, meta_shadow_window_id(window_id), meta_packet_idx, &meta_packets.at(i))) {
              std::abort();
            }
          } else if (!core_->tmem_read_meta_packet(handle, i, &meta_packets.at(i))) {
            std::abort();
          }
        }
        auto meta_bytes = copy_packets<MetaMem::packet_t>(meta_packets);
        if (!metamem_.fill_tile(slot_id, meta_bytes)) {
          std::abort();
        }
      }
    } break;
    case TcuTarget::B: {
      auto packet_bytes = copy_packets<BMem::packet_t>(packets);
      if (!bmem_.fill_tile(slot_id, slot.fmt_b, packet_bytes, slot.sparse_mode)) {
        std::abort();
      }
    } break;
    case TcuTarget::C: {
      auto packet_bytes = copy_packets<CMem::packet_t>(packets);
      if (!cmem_.fill_tile(slot_id, slot.fmt_c, packet_bytes)) {
        std::abort();
      }
    } break;
    default:
      std::abort();
    }
  }

  void enqueue_async_mma_load(uint32_t wid, uint32_t handle, const IntrTcuArgs& args, ExeTraceData* trace_data) {
    if (!core_->tmem_handle_ready_for_mma_load(handle, args.target, args.sparse_mode)) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    if (args.slot_id >= operand_slots_.size()) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    auto slot_id = static_cast<int32_t>(args.slot_id);
    auto& slot = operand_slots_.at(slot_id);
    if (!core_->ensure_tmem_window_bound(handle, args.descriptor, args.target, args.window_id)) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    if (!slot.valid || slot.owner_wid != wid || slot.descriptor != args.descriptor) {
      if (!slot_can_rebind(slot)) {
        if (trace_data) {
          trace_data->retry = true;
        }
        return;
      }
      init_slot_for_descriptor(static_cast<uint32_t>(slot_id), slot, wid, args);
    } else if (args.target == TcuTarget::None) {
      if (slot_has_pending_work(slot) || slot.c_dirty) {
        if (trace_data) {
          trace_data->retry = true;
        }
        return;
      }
      init_slot_for_descriptor(static_cast<uint32_t>(slot_id), slot, wid, args);
    } else if (!slot_target_available(slot, args.target)) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }

    auto async_id = core_->mma_load_async_issue(wid, handle, args.descriptor);
    uint32_t num_uops = 0;
    auto push_fill = [&](MemUop::Kind kind, TcuTarget target, bool separate_handle) {
      MemUop op{};
      op.kind = kind;
      op.wid = wid;
      op.slot_id = static_cast<uint32_t>(slot_id);
      op.handle = handle;
      op.window_id = args.window_id;
      op.async_id = async_id;
      op.separate_handle = separate_handle;
      WindowCursorRole cursor_role = WindowCursorRole::LoadA;
      switch (target) {
      case TcuTarget::A:
        cursor_role = WindowCursorRole::LoadA;
        break;
      case TcuTarget::B:
        cursor_role = WindowCursorRole::LoadB;
        break;
      case TcuTarget::C:
        cursor_role = WindowCursorRole::LoadC;
        break;
      default:
        break;
      }
      (void)reserve_window_cursor(handle, args.window_id, cursor_role, &op.tile_idx, &op.packets_per_tile);
      switch (target) {
      case TcuTarget::A:
        op.remaining_tmem_reads = a_packet_count(slot.fmt_a)
                                + ((slot.sparse_mode != vt::sparse_none) ? meta_packet_count(slot.sparse_mode) : 0);
        op.remaining_amem_writes = AMem::fill_beats();
        op.remaining_meta_writes = (slot.sparse_mode != vt::sparse_none) ? MetaMem::fill_beats() : 0;
        break;
      case TcuTarget::B:
        op.remaining_tmem_reads = b_packet_count(slot.fmt_b, slot.sparse_mode);
        op.remaining_bmem_writes = BMem::fill_beats(slot.sparse_mode);
        break;
      case TcuTarget::C:
        op.remaining_tmem_reads = c_packet_count(slot.fmt_c);
        op.remaining_cmem_writes = CMem::fill_beats(slot.fmt_c);
        break;
      default:
        std::abort();
      }
      mark_target_pending(slot, target, true);
      mem_ops_.push_back(op);
      perf_stats_.mem_queue_max = std::max<uint64_t>(perf_stats_.mem_queue_max, mem_ops_.size());
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

    pending_mem_ops_[async_id] = num_uops;
  }

  void enqueue_async_mma_store(uint32_t wid, uint32_t handle, const IntrTcuArgs& args, ExeTraceData* trace_data) {
    if (!core_->tmem_handle_ready_for_mma_store(handle)) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    if (args.slot_id >= operand_slots_.size()) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    auto slot_id = static_cast<int32_t>(args.slot_id);
    auto& slot = operand_slots_.at(slot_id);
    if (!core_->ensure_tmem_window_bound(handle, args.descriptor, TcuTarget::C, args.window_id)) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    if (args.target != TcuTarget::None && args.target != TcuTarget::C) {
      std::abort();
    }
    if (!slot.valid || slot.owner_wid != wid || slot.descriptor != args.descriptor
     || slot.store_pending || slot.c_wmma_inflight != 0) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    if (!slot.cmem_final_valid) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    auto& store_slot = operand_slots_.at(slot_id);
    store_slot.store_pending = true;
    store_slot.busy = true;
    store_slot.store_async_id = core_->mma_store_async_issue(wid, handle, args.descriptor);
    MemUop op{};
    op.kind = MemUop::Kind::StoreC;
    op.wid = wid;
    op.slot_id = static_cast<uint32_t>(slot_id);
    op.handle = handle;
    op.window_id = args.window_id;
    op.async_id = store_slot.store_async_id;
    op.separate_handle = args.target != TcuTarget::None;
    (void)reserve_window_cursor(handle, args.window_id, WindowCursorRole::StoreC, &op.tile_idx, &op.packets_per_tile);
    op.remaining_cmem_reads = CMem::dump_beats(store_slot.fmt_c);
    op.remaining_tmem_writes = c_packet_count(store_slot.fmt_c);
    mem_ops_.push_back(op);
    perf_stats_.mem_queue_max = std::max<uint64_t>(perf_stats_.mem_queue_max, mem_ops_.size());
  }

  void enqueue_async_wmma(uint32_t wid,
                          const IntrTcuArgs& args,
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
    if (args.slot_id >= operand_slots_.size() || args.c_slot_id >= operand_slots_.size()) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    auto ab_slot_id = static_cast<int32_t>(args.slot_id);
    auto c_slot_id = static_cast<int32_t>(args.c_slot_id);
    auto& ab_slot = operand_slots_.at(ab_slot_id);
    auto& c_slot = operand_slots_.at(c_slot_id);
    if (!ab_slot.valid || ab_slot.owner_wid != wid || ab_slot.descriptor != args.descriptor
     || !c_slot.valid || c_slot.owner_wid != wid || c_slot.descriptor != args.descriptor
     || !(ab_slot.a_ready && ab_slot.b_ready && c_slot.c_ready)
     || ab_slot.wmma_pending || c_slot.store_pending) {
      if (trace_data) {
        trace_data->retry = true;
      }
      return;
    }
    ab_slot.wmma_pending = true;
    ab_slot.busy = slot_has_pending_work(ab_slot);
    c_slot.busy = true;
    c_slot.cmem_final_valid = false;
    c_slot.c_dirty = true;
    ++c_slot.c_wmma_inflight;
    auto async_id = core_->wmma_async_issue(wid);
    ab_slot.wmma_async_id = async_id;
    pending_wmma_uops_[async_id] = kWmmaPrimitiveCount;
    pending_wmma_jobs_.push_back(PendingWmmaJob{
      wid,
      static_cast<uint32_t>(ab_slot_id),
      static_cast<uint32_t>(c_slot_id),
      fmt_a,
      fmt_b,
      fmt_c,
      args.sparse_mode,
      async_id,
      0,
    });
    ++perf_stats_.issued_macro_wmma;
    perf_stats_.pending_wmma_jobs_max = std::max<uint64_t>(perf_stats_.pending_wmma_jobs_max, pending_wmma_jobs_.size() + (active_wmma_job_valid_ ? 1 : 0));
  }

  void service_mem_ops() {
    if (mem_ops_.empty()) {
      return;
    }
    ensure_mem_port_budgets();
    for (size_t idx = 0; idx < mem_ops_.size(); ++idx) {
      auto& uop = mem_ops_.at(idx);
      auto& slot = operand_slots_.at(uop.slot_id);
      bool progressed = false;

      if (uop.kind == MemUop::Kind::FillA
       || uop.kind == MemUop::Kind::FillB
       || uop.kind == MemUop::Kind::FillC) {
        auto fill_action = select_fill_local_write(uop, slot);
        bool local_blocked = false;

        switch (fill_action) {
        case LocalFillAction::AData:
          if (amem_write_budget_ != 0) {
            --amem_write_budget_;
            progressed = perform_fill_local_write(uop, slot, fill_action);
          } else {
            local_blocked = true;
          }
          break;
        case LocalFillAction::BData:
          if (bmem_write_budget_ != 0) {
            --bmem_write_budget_;
            progressed = perform_fill_local_write(uop, slot, fill_action);
          } else {
            local_blocked = true;
          }
          break;
        case LocalFillAction::CData:
          if (cmem_write_budget_ != 0) {
            --cmem_write_budget_;
            progressed = perform_fill_local_write(uop, slot, fill_action);
          } else {
            local_blocked = true;
          }
          break;
        case LocalFillAction::Meta:
          if (meta_write_budget_ != 0) {
            --meta_write_budget_;
            progressed = perform_fill_local_write(uop, slot, fill_action);
          } else {
            local_blocked = true;
          }
          break;
        case LocalFillAction::None:
          break;
        }

        if (!progressed && uop.remaining_tmem_reads != 0) {
          if (try_acquire_fill_read_port(uop, slot)) {
            progressed = stage_fill_read(uop, slot);
          } else if (!local_blocked) {
            ++perf_stats_.stall_tmem_read_port_busy;
            continue;
          }
        }

        if (!progressed && local_blocked) {
          switch (fill_action) {
          case LocalFillAction::AData:
            ++perf_stats_.stall_amem_port_busy;
            break;
          case LocalFillAction::BData:
            ++perf_stats_.stall_bmem_port_busy;
            break;
          case LocalFillAction::CData:
            ++perf_stats_.stall_cmem_port_busy;
            break;
          case LocalFillAction::Meta:
            ++perf_stats_.stall_meta_port_busy;
            break;
          case LocalFillAction::None:
            break;
          }
          continue;
        }
      } else if (uop.kind == MemUop::Kind::StoreC) {
        if (uop.staged_store_packet_cursor < uop.staged_store_packets.size()) {
          if (try_acquire_store_write_port(uop, slot)) {
            progressed = emit_store_packet(uop, slot);
          } else {
            ++perf_stats_.stall_tmem_write_port_busy;
            continue;
          }
        } else if (uop.remaining_cmem_reads != 0) {
          if (cmem_read_budget_ == 0) {
            ++perf_stats_.stall_cmem_port_busy;
            continue;
          }
          --cmem_read_budget_;
          progressed = stage_store_read(uop, slot);
        }
      } else if (uop.remaining_amem_writes != 0 || uop.remaining_bmem_writes != 0
              || uop.remaining_cmem_writes != 0 || uop.remaining_cmem_reads != 0
              || uop.remaining_meta_writes != 0) {
        if (uop.remaining_amem_writes != 0) {
          if (amem_write_budget_ == 0) {
            ++perf_stats_.stall_amem_port_busy;
            continue;
          }
          --amem_write_budget_;
          --uop.remaining_amem_writes;
          progressed = true;
        }
        if (uop.remaining_bmem_writes != 0) {
          if (bmem_write_budget_ == 0) {
            ++perf_stats_.stall_bmem_port_busy;
            continue;
          }
          --bmem_write_budget_;
          --uop.remaining_bmem_writes;
          progressed = true;
        }
        if (uop.remaining_cmem_writes != 0) {
          if (cmem_write_budget_ == 0) {
            ++perf_stats_.stall_cmem_port_busy;
            continue;
          }
          --cmem_write_budget_;
          --uop.remaining_cmem_writes;
          progressed = true;
        }
        if (uop.remaining_cmem_reads != 0) {
          if (cmem_read_budget_ == 0) {
            ++perf_stats_.stall_cmem_port_busy;
            continue;
          }
          --cmem_read_budget_;
          --uop.remaining_cmem_reads;
          progressed = true;
        }
        if (uop.remaining_meta_writes != 0) {
          if (meta_write_budget_ == 0) {
            ++perf_stats_.stall_meta_port_busy;
            continue;
          }
          --meta_write_budget_;
          --uop.remaining_meta_writes;
          progressed = true;
        }
      } else if (uop.remaining_tmem_writes != 0) {
        if (core_->try_acquire_tmem_write_port(uop.handle, uop.next_store_packet_idx)) {
          --uop.remaining_tmem_writes;
          progressed = true;
        } else {
          ++perf_stats_.stall_tmem_write_port_busy;
          continue;
        }
      }

      if (!progressed || !mem_op_complete(uop)) {
        continue;
      }

      switch (uop.kind) {
      case MemUop::Kind::FillA:
        mark_target_ready(slot, TcuTarget::A);
        break;
      case MemUop::Kind::FillB:
        mark_target_ready(slot, TcuTarget::B);
        break;
      case MemUop::Kind::FillC:
        mark_target_ready(slot, TcuTarget::C);
        break;
      case MemUop::Kind::StoreC: {
        slot.store_pending = false;
        slot.c_dirty = false;
        slot.cmem_final_valid = true;
        slot.busy = slot_has_pending_work(slot);
        core_->async_tensor_complete(uop.async_id);
        mem_ops_.erase(mem_ops_.begin() + idx);
        return;
      }
      default:
        std::abort();
      }

      auto pending_it = pending_mem_ops_.find(uop.async_id);
      if (pending_it != pending_mem_ops_.end()) {
        if (pending_it->second == 0) {
          std::abort();
        }
        --pending_it->second;
        if (pending_it->second == 0) {
          pending_mem_ops_.erase(pending_it);
          core_->async_tensor_complete(uop.async_id);
        }
      }
      mem_ops_.erase(mem_ops_.begin() + idx);
      return;
    }
  }

  void dispatch_compute_uop() {
    bool has_work = active_wmma_job_valid_ || !pending_wmma_jobs_.empty();
    if (!tensorcore_.ready(true)) {
      if (has_work) {
        ++perf_stats_.stall_tc_busy;
      }
      return;
    }
    if (!active_wmma_job_valid_) {
      if (pending_wmma_jobs_.empty()) {
        bool tensor_frontend_live = !mem_ops_.empty() || !pending_wmma_uops_.empty();
        if (!tensor_frontend_live) {
          for (const auto& slot : operand_slots_) {
            tensor_frontend_live |= slot.valid || slot.c_dirty || slot_has_pending_work(slot);
          }
        }
        if (tensor_frontend_live) {
          record_no_wmma_ready_stall();
        }
        return;
      }
      active_wmma_job_ = pending_wmma_jobs_.front();
      pending_wmma_jobs_.pop_front();
      active_wmma_job_valid_ = true;
    }

    auto& job = active_wmma_job_;
    auto& ab_slot = operand_slots_.at(job.ab_slot_id);

    uint32_t storage_k = (job.next_uop < kSubtilesPerTile) ? 0 : 1;
    uint32_t c_subtile_id = job.next_uop & (kSubtilesPerTile - 1);
    uint32_t storage_m = c_subtile_id / 2;
    uint32_t storage_n = c_subtile_id % 2;
    uint16_t a_block[kPrimitiveDim][kPrimitiveDim] = {};
    uint16_t b_block[kPrimitiveDim][kPrimitiveDim] = {};
    uint32_t zero_c[kPrimitiveDim][kPrimitiveDim] = {};

    amem_.read_primitive(job.ab_slot_id, storage_m, storage_k, a_block);
    // Sparse B is temporarily materialized through the dense 576b layout.
    bmem_.read_primitive(job.ab_slot_id, storage_k, storage_n, b_block);

    TensorCoreMeta meta{};
    meta.wid = job.wid;
    meta.async_id = job.async_id;
    meta.ab_slot_id = job.ab_slot_id;
    meta.c_slot_id = job.c_slot_id;
    meta.c_subtile_id = c_subtile_id;
    meta.valid = true;

    tensorcore_fmt_c_ = job.fmt_c;
    configure_open_tensorcore_precision(job.fmt_c);
    tensorcore_.push_uop(a_block, b_block, zero_c, meta);
    auto cycle = core_->current_cycle();
    ++perf_stats_.issued_primitive_tiles;
    if (perf_stats_.first_tc_issue_cycle == 0) {
      perf_stats_.first_tc_issue_cycle = cycle;
    }
    perf_stats_.last_tc_issue_cycle = cycle;

    ++job.next_uop;
    if (job.next_uop >= kWmmaPrimitiveCount) {
      ab_slot.wmma_pending = false;
      ab_slot.a_ready = false;
      ab_slot.busy = slot_has_pending_work(ab_slot);
      active_wmma_job_valid_ = false;
      return;
    }
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
    auto& slot = operand_slots_.at(retire.meta.c_slot_id);
    auto subtile_id = retire.meta.c_subtile_id;

    auto cycle = core_->current_cycle();
    ++perf_stats_.retired_primitive_tiles;
    if (perf_stats_.first_tc_retire_cycle == 0) {
      perf_stats_.first_tc_retire_cycle = cycle;
    }
    perf_stats_.last_tc_retire_cycle = cycle;

    cmem_.accumulate_subtile(retire.meta.c_slot_id, subtile_id, retire.fp22_out);

    auto pending_it = pending_wmma_uops_.find(retire.meta.async_id);
    if (pending_it == pending_wmma_uops_.end() || pending_it->second == 0) {
      std::abort();
    }
    --pending_it->second;
    if (pending_it->second == 0) {
      pending_wmma_uops_.erase(pending_it);
      if (slot.c_wmma_inflight == 0) {
        std::abort();
      }
      --slot.c_wmma_inflight;
      slot.busy = slot_has_pending_work(slot);
      slot.c_dirty = true;
      slot.cmem_final_valid = true;
      ++perf_stats_.retired_macro_wmma;
      core_->async_tensor_complete(retire.meta.async_id);
    }
  }

  TensorUnit* simobject_;
  Core*       core_;
  Arch        arch_;
  PerfStats   perf_stats_;
  std::array<OperandSlot, kNumOperandSlots> operand_slots_;
  AMem amem_;
  BMem bmem_;
  CMem cmem_;
  MetaMem metamem_;
  std::deque<MemUop> mem_ops_;
  std::unordered_map<uint32_t, uint32_t> pending_mem_ops_;
  std::unordered_map<uint32_t, uint32_t> pending_wmma_uops_;
  std::deque<PendingWmmaJob> pending_wmma_jobs_;
  PendingWmmaJob active_wmma_job_;
  bool active_wmma_job_valid_ = false;
  TensorCoreTop tensorcore_;
  uint32_t tensorcore_fmt_c_ = vt::fp16::id;
  std::unordered_map<WindowCursorKey, WindowCursorState, WindowCursorKeyHash> window_cursors_;
  uint64_t mem_port_cycle_ = std::numeric_limits<uint64_t>::max();
  uint32_t amem_write_budget_ = 0;
  uint32_t bmem_write_budget_ = 0;
  uint32_t cmem_write_budget_ = 0;
  uint32_t cmem_read_budget_ = 0;
  uint32_t meta_write_budget_ = 0;
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
      return {"MMA_LOAD.ASYNC." + std::to_string(args.descriptor) + "." + std::to_string(args.slot_id)
            + "." + std::to_string(static_cast<uint32_t>(args.target)), ""};
    }
    return {"MMA_LOAD." + std::string(vt::fmt_string(fmt_a)) + "." + std::string(vt::fmt_string(fmt_b)) + "." + std::string(vt::fmt_string(args.fmt_c))
          + "." + std::to_string(static_cast<uint32_t>(args.target)), ""};
  case TcuType::MMA_STORE:
    if (args.macro_op) {
      return {"MMA_STORE.ASYNC." + std::to_string(args.descriptor) + "." + std::to_string(args.slot_id)
            + "." + std::to_string(static_cast<uint32_t>(args.target)), ""};
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
      return {"WMMA.ASYNC." + std::to_string(args.descriptor) + "." + std::to_string(args.slot_id)
            + "." + std::to_string(args.c_slot_id), ""};
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

TensorUnit::IssueBlockReason TensorUnit::classify_issue_block(uint32_t wid, TcuType tcu_type, IntrTcuArgs args) const {
  return impl_->classify_issue_block(wid, tcu_type, args);
}

void TensorUnit::record_issue_stall(IssueBlockReason reason) {
  impl_->record_issue_stall(reason);
}

void TensorUnit::record_no_tensor_instr_candidate_stall() {
  impl_->record_issue_stall(IssueBlockReason::NoTensorInstrCandidate);
}

const TensorUnit::PerfStats& TensorUnit::perf_stats() const {
  return impl_->perf_stats();
}
