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

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace vortex {

enum class TmemWindowTarget : uint8_t {
  A = 0,
  B,
  C,
  D,
  Meta,
};

enum class TmemWindowLayoutKind : uint8_t {
  LinearPacketStream = 0,
  MathRowMajor,
  ALineNative,
  BLineNative,
  CSubtileNative,
};

struct TensorShape2D {
  uint16_t rows = 0;
  uint16_t cols = 0;

  bool empty() const {
    return rows == 0 || cols == 0;
  }
};

struct TmemWindowPlannerInput {
  TensorShape2D a_shape = {};
  TensorShape2D b_shape = {};
  TensorShape2D c_shape = {};
  TensorShape2D d_shape = {};
  TensorShape2D meta_shape = {};
  uint32_t fmt_a = 0;
  uint32_t fmt_b = 0;
  uint32_t fmt_c = 0;
  uint32_t fmt_d = 0;
  uint32_t sparse_mode = 0;
  uint32_t allocation_col_span = 0;
};

struct TmemWindowPlan {
  uint32_t window_id = 0;
  TmemWindowTarget target = TmemWindowTarget::A;
  TmemWindowLayoutKind layout_kind = TmemWindowLayoutKind::LinearPacketStream;
  TensorShape2D elem_shape = {};
  uint32_t fmt = 0;
  uint32_t sparse_mode = 0;


  // 逻辑参数 (基于元素的映射),TMEM_SHIFT和MMA_STORE/MMA_LOAD都是基于逻辑坐标, 由 layout_kind 决定含义
  uint32_t logical_col_base = 0;
  uint32_t logical_line_base = 0;
  uint32_t logical_col_span = 0;
  uint32_t logical_line_span = 0;
  uint32_t logical_tile_col_span = 0;
  uint32_t logical_tile_line_span = 0;
  uint32_t logical_packet_col_span = 0;
  uint32_t logical_packet_line_span = 0;

  // 物理映射参数 (基于 tile 的映射),主要用于MMA_LOAD进行搬运, 由 layout_kind 决定含义
  uint32_t packet_base = 0;
  uint32_t tile_rows = 0;
  uint32_t tile_cols = 0;
  uint32_t tile_count = 0;
  uint32_t packet_rows = 0;
  uint32_t packet_cols = 0;
  uint32_t packets_per_tile = 0;
};

struct TmemMathPacketRegion {
  uint32_t math_row_base = 0;
  uint32_t math_col_base = 0;
  uint32_t packet_rows = 0;
  uint32_t packet_cols = 0;
};

struct TmemLayoutPlan {
  bool valid = false;
  uint32_t epoch = 0;
  uint32_t required_col_span = 0;
  uint32_t required_logical_col_span = 0;
  uint32_t required_logical_line_span = 0;
  std::vector<TmemWindowPlan> windows;
};

class TmemWindowPlanner {
public:
  static constexpr uint32_t kMinAllocationCols = 16;
  static constexpr uint32_t kLogicalCols = 64;
  static constexpr uint32_t kLogicalLines = 128;
  static constexpr uint32_t kPacketBytes = 64;

  // m16n16k16 计算原语决定的固定 tile 大小
  static constexpr uint32_t kATileRows = 16;  // M 维
  static constexpr uint32_t kATileCols = 16;  // K 维
  static constexpr uint32_t kBTileRows = 16;  // K 维
  static constexpr uint32_t kBTileCols = 16;  // N 维
  static constexpr uint32_t kCDTileRows = 16; // M 维
  static constexpr uint32_t kCDTileCols = 16; // N 维

  // 兼容旧代码的别名（C/D tile 尺寸）
  static constexpr uint32_t kTileRows = kCDTileRows;
  static constexpr uint32_t kTileCols = kCDTileCols;

  // 根据 target 返回 tile 行数
  static constexpr uint32_t tile_rows_for(TmemWindowTarget t) {
    switch (t) {
    case TmemWindowTarget::A:    return kATileRows;
    case TmemWindowTarget::B:    return kBTileRows;
    default:                     return kCDTileRows;
    }
  }

  // 根据 target 返回 tile 列数
  static constexpr uint32_t tile_cols_for(TmemWindowTarget t) {
    switch (t) {
    case TmemWindowTarget::A:    return kATileCols;
    case TmemWindowTarget::B:    return kBTileCols;
    default:                     return kCDTileCols;
    }
  }

  // ---- Window shape 与输出驻留 (output_resident) 是正交概念 ----
  //
  // Window shape 决定 TMA 搬运粒度 (MMA descriptor 的 a_rows/a_cols 等字段):
  //   A = M × K,  B = K × N,  C = D = M × N
  //
  // 输出驻留 (output_resident!=0) 决定 WMMA 结果写 CMem(FIFO)累加还是写 DMem 独立输出。
  // 任何 window shape 都可以搭配 output_resident=0 或 output_resident=1, 由 MMA_LOAD 的 tile_id
  // 显式控制计算哪个 tile, 软件决定迭代顺序。
  //
  // 支持的 window shapes (col_span=64, fp8 A/B + fp16 C/D):
  //
  //   M×N×K        A window   B window   C/D window   lines   tiles(A/B/C)
  //   ─────────────────────────────────────────────────────────────────────
  //   16×16×16     16×16      16×16      16×16        24/128  1/1/1
  //   32×32×16     32×16      16×32      32×32        80/128  2/2/4
  //   32×32×32     32×32      32×32      32×32        96/128  4/4/4

  // 验证 MMA descriptor 的 shapes 数学一致性
  // A(M×K) × B(K×N) = C(M×N), D shape = C shape
  // M 须是 16 的倍数, N 须是 16 的倍数, K 须是 16 的倍数
  static bool validate_mma_shapes(const TmemWindowPlannerInput& input,
                                  std::string* reason = nullptr);

  // 计算 allocation 所需的最小 col_span
  static uint32_t compute_min_col_span(const TmemWindowPlannerInput& input);

  static bool build_dense_plan(const TmemWindowPlannerInput& input,
                               TmemLayoutPlan* out,
                               std::string* reason = nullptr);
  static bool build_single_dense_window(TmemWindowTarget target,
                                        const TensorShape2D& shape,
                                        uint32_t fmt,
                                        uint32_t sparse_mode,
                                        uint32_t allocation_col_span,
                                        uint32_t window_id,
                                        TmemWindowPlan* out,
                                        std::string* reason = nullptr);

  static bool uses_math_packet_adapter(const TmemWindowPlan& window);
  static bool packet_math_region(const TmemWindowPlan& window,
                                 uint32_t packet_idx,
                                 TmemMathPacketRegion* out);
  static bool pack_math_packet(const TmemWindowPlan& window,
                               const std::vector<uint8_t>& payload,
                               uint32_t packet_idx,
                               std::array<uint8_t, kPacketBytes>* out,
                               bool transpose = false);
  static bool unpack_math_packet(const TmemWindowPlan& window,
                                 uint32_t packet_idx,
                                 const std::array<uint8_t, kPacketBytes>& packet,
                                 std::vector<uint8_t>* payload);

private:
  static bool append_dense_window(TmemLayoutPlan* plan,
                                  uint32_t* next_window_id,
                                  uint32_t* required_alloc_col_span,  // 同时作为输入 col_span 和输出 max
                                  uint32_t* next_line_base,
                                  TmemWindowTarget target,
                                  const TensorShape2D& shape,
                                  uint32_t fmt,
                                  uint32_t sparse_mode,
                                  std::string* reason);
};

} // namespace vortex
