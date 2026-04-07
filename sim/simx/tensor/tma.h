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

#include <cstdint>
#include <functional>
#include <vector>
#include "tmem.h"
#include "types.h"

namespace vortex {

// Software-visible TMA transfer descriptor.
//
// The descriptor describes one external-memory matrix window and the logical
// TMEM destination/source footprint used by the Core-side TMA engine.
struct TmaDescriptor {
  uint64_t addr = 0;
  uint32_t size_bytes = 0;
  uint32_t stride_bytes = 0;
  uint16_t rows = 0;
  uint16_t cols = 0;
  uint16_t elem_bytes = 0;
  uint16_t flags = 0;
  uint64_t meta_addr = 0;
  uint32_t meta_size_bytes = 0;
  uint16_t tmem_base = 0;
  uint16_t meta_tmem_base = 0;
  // Legacy field name retained for ABI compatibility. In the current model it
  // denotes the allocation span in logical TMEM columns, not physical banks.
  uint16_t bank_span = 0;
  uint16_t meta_col_span = 0;
  uint8_t tile_role = static_cast<uint8_t>(TcuTarget::None);
  uint8_t payload_kind = static_cast<uint8_t>(TcuPayloadKind::Dense);
  uint8_t transpose = 0;     // TMA 以转置寻址模式从 DRAM 读取
  uint8_t reserved = 0;
} __attribute__((packed));

// Software-visible MMA descriptor.
//
// Shapes are expressed in mathematical matrix elements (window shape).
// The window planner translates these into TMEM windows, tiles and packets.
// transpose 已移至 TmaDescriptor (数据搬运侧关注)。
struct MmaDescriptor {
  uint32_t fmt_a = 0;
  uint32_t fmt_b = 0;
  uint32_t fmt_c = 0;
  uint32_t fmt_d = 0;
  uint8_t ws = 0;
  uint8_t sp = 0;
  uint8_t sparse_mode = 0;
  uint8_t reserved[5] = {};
  uint16_t a_rows = 0;
  uint16_t a_cols = 0;
  uint16_t b_rows = 0;
  uint16_t b_cols = 0;
  uint16_t c_rows = 0;
  uint16_t c_cols = 0;
} __attribute__((packed));

// Front-end TMA timing/descriptor helper.
//
// Hardware view:
// - fetch descriptor tables from device memory
// - estimate launch/setup latency for a TMA transaction
// - marshal row-major payload / metadata bytes between device memory and the
//   packet-sized reorder buffers that feed TMEM ingress
//
// Per-cycle TMEM packet progression happens later in Core once an
// AsyncTensorOp has been issued.
class TmaModel {
public:
  using ReadCallback = std::function<void(void*, uint64_t, uint32_t)>;
  using WriteCallback = std::function<void(const void*, uint64_t, uint32_t)>;

  static constexpr uint32_t kLoadBaseLatency = 12;   // descriptor fetch + addr xlat + pipeline setup
  static constexpr uint32_t kLoadBytesPerCycle = 64;
  static constexpr uint32_t kStoreBaseLatency = 12;
  static constexpr uint32_t kStoreBytesPerCycle = 64;
  static constexpr uint32_t kTransposePenalty = 8;

  explicit TmaModel(bool realistic_load = true);

  void reset();

  bool read_tma_descriptor(uint64_t startup_arg,
                           uint32_t desc_id,
                           TmaDescriptor* out,
                           const ReadCallback& dcache_read);

  bool read_mma_descriptor(uint64_t startup_arg,
                           uint32_t desc_id,
                           MmaDescriptor* out,
                           const ReadCallback& dcache_read);

  uint32_t estimate_load_latency(const TmaDescriptor& desc) const;
  uint32_t payload_size_bytes(const TmaDescriptor& desc) const;
  uint32_t payload_packet_count(const TmaDescriptor& desc) const;
  uint32_t meta_packet_count(const TmaDescriptor& desc) const;

  bool load_linear_packet(uint64_t base_addr,
                          uint32_t size_bytes,
                          uint32_t packet_idx,
                          TmemPacket* out,
                          const ReadCallback& dcache_read) const;

  void store_linear_packet(uint64_t base_addr,
                           uint32_t size_bytes,
                           uint32_t packet_idx,
                           const TmemPacket& packet,
                           const WriteCallback& dcache_write) const;

  bool load_math_packet(uint64_t base_addr,
                        uint32_t row_stride_bytes,
                        uint32_t rows,
                        uint32_t cols,
                        uint32_t elem_bytes,
                        const TmemWindowPlan& window,
                        uint32_t packet_idx,
                        TmemPacket* out,
                        const ReadCallback& dcache_read,
                        bool transpose = false) const;

  void store_math_packet(uint64_t base_addr,
                         uint32_t row_stride_bytes,
                         uint32_t rows,
                         uint32_t cols,
                         uint32_t elem_bytes,
                         const TmemWindowPlan& window,
                         uint32_t packet_idx,
                         const TmemPacket& packet,
                         const WriteCallback& dcache_write) const;

  bool load_payload_packet(const TmaDescriptor& desc,
                           uint32_t payload_size_bytes,
                           const TmemWindowPlan* payload_window,
                           uint32_t packet_idx,
                           TmemPacket* out,
                           const ReadCallback& dcache_read) const;

  void store_payload_packet(const TmaDescriptor& desc,
                            uint32_t payload_size_bytes,
                            const TmemWindowPlan* payload_window,
                            uint32_t packet_idx,
                            const TmemPacket& packet,
                            const WriteCallback& dcache_write) const;

  bool load_meta_packet(const TmaDescriptor& desc,
                        uint32_t meta_size_bytes,
                        const TmemWindowPlan* meta_window,
                        uint32_t packet_idx,
                        TmemPacket* out,
                        const ReadCallback& dcache_read) const;

  bool load_row_chunk_packet(uint64_t base_addr,
                             uint32_t row_stride_bytes,
                             uint32_t row_idx,
                             uint32_t chunk_idx,
                             uint32_t row_bytes,
                             TmemPacket* out,
                             const ReadCallback& dcache_read) const;

  bool load_payload(const TmaDescriptor& desc,
                    uint32_t capacity,
                    std::vector<uint8_t>* out,
                    const ReadCallback& dcache_read) const;

  bool load_meta(const TmaDescriptor& desc,
                 uint32_t capacity,
                 std::vector<uint8_t>* out,
                 const ReadCallback& dcache_read) const;

  void store_payload(const TmaDescriptor& desc,
                     const uint8_t* data,
                     uint32_t size_bytes,
                     const WriteCallback& dcache_write) const;

private:
  bool ensure_descriptor_tables_loaded(uint64_t startup_arg, const ReadCallback& dcache_read);

  bool descriptor_tables_loaded_;
  bool realistic_load_;
  std::vector<TmaDescriptor> tma_desc_table_;
  std::vector<MmaDescriptor> mma_desc_table_;
};

} // namespace vortex
