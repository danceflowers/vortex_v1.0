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

#include <algorithm>
#include <vector>
#include <tensor_cfg.h>
#include "tma.h"
#include "tmem.h"

using namespace vortex;

namespace {

uint32_t fmt_bytes(uint32_t fmt) {
  switch (fmt) {
  case vortex::tensor::fp8::id:
  case vortex::tensor::uint8::id:
    return 1;
  case vortex::tensor::fp16::id:
  case vortex::tensor::bf16::id:
    return 2;
  case vortex::tensor::fp32::id:
    return 4;
  default:
    return 0;
  }
}

} // namespace

// TmaModel is the descriptor/launch-side front-end of the tensor memory
// accelerator. It translates software-visible descriptors into launch latency
// estimates plus row-major packet/line-chunk accesses against external memory.
// TmaFrontend later uses these helpers to build the small packet-sized reorder
// buffers that feed TMEM ingress.

TmaModel::TmaModel(bool realistic_load)
  : descriptor_tables_loaded_(false)
  , realistic_load_(realistic_load) {
  (void)realistic_load_;
}

void TmaModel::reset() {
  descriptor_tables_loaded_ = false;
  tma_desc_table_.clear();
  mma_desc_table_.clear();
}

bool TmaModel::ensure_descriptor_tables_loaded(uint64_t startup_arg, const ReadCallback& dcache_read) {
  if (descriptor_tables_loaded_) {
    return true;
  }
  if (0 == startup_arg || !dcache_read) {
    return false;
  }

  vortex::tensor::descriptor_table_arg_t header;
  dcache_read(&header, startup_arg, sizeof(header));
  if (header.magic != vortex::tensor::descriptor_table_magic
   || header.version != vortex::tensor::descriptor_table_version) {
    return false;
  }

  tma_desc_table_.clear();
  mma_desc_table_.clear();
  if (header.tma_desc_count != 0) {
    tma_desc_table_.resize(header.tma_desc_count);
    dcache_read(tma_desc_table_.data(),
                header.tma_desc_addr,
                header.tma_desc_count * sizeof(TmaDescriptor));
  }
  if (header.mma_desc_count != 0) {
    mma_desc_table_.resize(header.mma_desc_count);
    dcache_read(mma_desc_table_.data(),
                header.mma_desc_addr,
                header.mma_desc_count * sizeof(IDescriptor));
  }
  descriptor_tables_loaded_ = true;
  return true;
}

bool TmaModel::read_tma_descriptor(uint64_t startup_arg,
                                   uint32_t desc_id,
                                   TmaDescriptor* out,
                                   const ReadCallback& dcache_read) {
  if (nullptr == out || !ensure_descriptor_tables_loaded(startup_arg, dcache_read) || desc_id >= tma_desc_table_.size()) {
    return false;
  }
  *out = tma_desc_table_.at(desc_id);
  return true;
}

bool TmaModel::read_idescriptor(uint64_t startup_arg,
                                   uint32_t desc_id,
                                   IDescriptor* out,
                                   const ReadCallback& dcache_read) {
  if (nullptr == out || !ensure_descriptor_tables_loaded(startup_arg, dcache_read) || desc_id >= mma_desc_table_.size()) {
    return false;
  }
  *out = mma_desc_table_.at(desc_id);
  return true;
}

uint32_t TmaModel::estimate_load_latency(const TmaDescriptor& desc) const {
  uint32_t size_bytes = payload_size_bytes(desc);
  if (desc.meta_addr != 0 && desc.meta_size_bytes != 0 && desc.meta_col_span != 0) {
    size_bytes += desc.meta_size_bytes;
  }
  uint32_t transfer_cycles = std::max<uint32_t>(1, (size_bytes + kLoadBytesPerCycle - 1) / kLoadBytesPerCycle);
  return kLoadBaseLatency + transfer_cycles;
}

uint32_t TmaModel::payload_size_bytes(const TmaDescriptor& desc) const {
  uint32_t matrix_bytes = desc.rows * desc.cols * desc.elem_bytes;
  uint32_t requested_size = desc.size_bytes ? desc.size_bytes : matrix_bytes;
  return requested_size ? requested_size : matrix_bytes;
}

uint32_t TmaModel::payload_packet_count(const TmaDescriptor& desc) const {
  auto size_bytes = payload_size_bytes(desc);
  return (size_bytes + Tmem::kPacketBytes - 1) / Tmem::kPacketBytes;
}

uint32_t TmaModel::meta_packet_count(const TmaDescriptor& desc) const {
  if (desc.meta_addr == 0 || desc.meta_size_bytes == 0 || desc.meta_col_span == 0) {
    return 0;
  }
  return (desc.meta_size_bytes + Tmem::kPacketBytes - 1) / Tmem::kPacketBytes;
}

bool TmaModel::load_linear_packet(uint64_t base_addr,
                                  uint32_t size_bytes,
                                  uint32_t packet_idx,
                                  TmemPacket* out,
                                  const ReadCallback& dcache_read) const {
  if (nullptr == out || !dcache_read) {
    return false;
  }
  out->bytes.fill(0);
  auto byte_offset = packet_idx * Tmem::kPacketBytes;
  if (byte_offset >= size_bytes) {
    return true;
  }
  auto valid_bytes = std::min<uint32_t>(Tmem::kPacketBytes, size_bytes - byte_offset);
  dcache_read(out->bytes.data(), base_addr + byte_offset, valid_bytes);
  return true;
}

void TmaModel::store_linear_packet(uint64_t base_addr,
                                   uint32_t size_bytes,
                                   uint32_t packet_idx,
                                   const TmemPacket& packet,
                                   const WriteCallback& dcache_write) const {
  if (!dcache_write) {
    return;
  }
  auto byte_offset = packet_idx * Tmem::kPacketBytes;
  if (byte_offset >= size_bytes) {
    return;
  }
  auto valid_bytes = std::min<uint32_t>(Tmem::kPacketBytes, size_bytes - byte_offset);
  dcache_write(packet.bytes.data(), base_addr + byte_offset, valid_bytes);
}

bool TmaModel::load_math_packet(uint64_t base_addr,
                                uint32_t row_stride_bytes,
                                uint32_t rows,
                                uint32_t cols,
                                uint32_t elem_bytes,
                                const TmemWindowPlan& window,
                                uint32_t packet_idx,
                                TmemPacket* out,
                                const ReadCallback& dcache_read,
                                bool transpose) const {
  if (nullptr == out || !dcache_read || elem_bytes == 0) {
    return false;
  }
  out->bytes.fill(0);
  TmemMathPacketRegion region{};
  if (!TmemWindowPlanner::packet_math_region(window, packet_idx, &region)) {
    return false;
  }
  if (row_stride_bytes == 0) {
    row_stride_bytes = cols * elem_bytes;
  }
  uint32_t packet_byte_offset = 0;
  std::array<uint8_t, Tmem::kPacketBytes> row_chunk{};
  for (uint32_t packet_row = 0; packet_row < region.packet_rows; ++packet_row) {
    auto math_row = region.math_row_base + packet_row;
    auto valid_cols = (math_row < rows && region.math_col_base < cols)
                    ? std::min<uint32_t>(region.packet_cols, cols - region.math_col_base)
                    : 0;
    auto valid_bytes = valid_cols * elem_bytes;
    if (valid_bytes != 0) {
      if (transpose) {
        // Cycle-accurate 转置: 逐元素 strided 读取
        // 目标 (math_row, math_col) ← 源 (math_col, math_row)
        // 源地址: base + src_row * stride + src_col * elem_bytes
        // 其中 src_row = math_col, src_col = math_row
        // 每个元素单独一次 dcache 请求 (strided access → 更多 cache miss)
        for (uint32_t pc = 0; pc < valid_cols; ++pc) {
          auto src_row = region.math_col_base + pc;  // 目标列 → 源行
          auto src_col = math_row;                    // 目标行 → 源列
          dcache_read(row_chunk.data() + pc * elem_bytes,
                      base_addr + static_cast<uint64_t>(src_row) * row_stride_bytes
                               + static_cast<uint64_t>(src_col) * elem_bytes,
                      elem_bytes);
        }
      } else {
        // 非转置: 连续读取一行 (cache-friendly)
        dcache_read(row_chunk.data(),
                    base_addr + static_cast<uint64_t>(math_row) * row_stride_bytes
                             + static_cast<uint64_t>(region.math_col_base) * elem_bytes,
                    valid_bytes);
      }
      std::copy_n(row_chunk.data(), valid_bytes, out->bytes.data() + packet_byte_offset);
    }
    packet_byte_offset += region.packet_cols * elem_bytes;
  }
  return true;
}

void TmaModel::store_math_packet(uint64_t base_addr,
                                 uint32_t row_stride_bytes,
                                 uint32_t rows,
                                 uint32_t cols,
                                 uint32_t elem_bytes,
                                 const TmemWindowPlan& window,
                                 uint32_t packet_idx,
                                 const TmemPacket& packet,
                                 const WriteCallback& dcache_write) const {
  if (!dcache_write || elem_bytes == 0) {
    return;
  }
  TmemMathPacketRegion region{};
  if (!TmemWindowPlanner::packet_math_region(window, packet_idx, &region)) {
    return;
  }
  if (row_stride_bytes == 0) {
    row_stride_bytes = cols * elem_bytes;
  }
  uint32_t packet_byte_offset = 0;
  for (uint32_t packet_row = 0; packet_row < region.packet_rows; ++packet_row) {
    auto math_row = region.math_row_base + packet_row;
    auto valid_cols = (math_row < rows && region.math_col_base < cols)
                    ? std::min<uint32_t>(region.packet_cols, cols - region.math_col_base)
                    : 0;
    auto valid_bytes = valid_cols * elem_bytes;
    if (valid_bytes != 0) {
      dcache_write(packet.bytes.data() + packet_byte_offset,
                   base_addr + static_cast<uint64_t>(math_row) * row_stride_bytes
                            + static_cast<uint64_t>(region.math_col_base) * elem_bytes,
                   valid_bytes);
    }
    packet_byte_offset += region.packet_cols * elem_bytes;
  }
}

bool TmaModel::load_payload_packet(const TmaDescriptor& desc,
                                   uint32_t payload_size_bytes,
                                   const TmemWindowPlan* payload_window,
                                   uint32_t packet_idx,
                                   TmemPacket* out,
                                   const ReadCallback& dcache_read) const {
  if (nullptr != payload_window && TmemWindowPlanner::uses_math_packet_adapter(*payload_window)) {
    return load_math_packet(desc.addr,
                            desc.stride_bytes,
                            desc.rows,
                            desc.cols,
                            desc.elem_bytes,
                            *payload_window,
                            packet_idx,
                            out,
                            dcache_read,
                            desc.transpose != 0);
  }
  return load_linear_packet(desc.addr, payload_size_bytes, packet_idx, out, dcache_read);
}

void TmaModel::store_payload_packet(const TmaDescriptor& desc,
                                    uint32_t payload_size_bytes,
                                    const TmemWindowPlan* payload_window,
                                    uint32_t packet_idx,
                                    const TmemPacket& packet,
                                    const WriteCallback& dcache_write) const {
  if (nullptr != payload_window && TmemWindowPlanner::uses_math_packet_adapter(*payload_window)) {
    store_math_packet(desc.addr,
                      desc.stride_bytes,
                      desc.rows,
                      desc.cols,
                      desc.elem_bytes,
                      *payload_window,
                      packet_idx,
                      packet,
                      dcache_write);
    return;
  }
  store_linear_packet(desc.addr, payload_size_bytes, packet_idx, packet, dcache_write);
}

bool TmaModel::load_meta_packet(const TmaDescriptor& desc,
                                uint32_t meta_size_bytes,
                                const TmemWindowPlan* meta_window,
                                uint32_t packet_idx,
                                TmemPacket* out,
                                const ReadCallback& dcache_read) const {
  if (nullptr == out || !dcache_read || desc.meta_addr == 0) {
    return false;
  }
  if (nullptr != meta_window && TmemWindowPlanner::uses_math_packet_adapter(*meta_window)) {
    auto elem_bytes = fmt_bytes(meta_window->fmt);
    return load_math_packet(desc.meta_addr,
                            meta_window->elem_shape.cols * elem_bytes,
                            meta_window->elem_shape.rows,
                            meta_window->elem_shape.cols,
                            elem_bytes,
                            *meta_window,
                            packet_idx,
                            out,
                            dcache_read);
  }
  return load_linear_packet(desc.meta_addr, meta_size_bytes, packet_idx, out, dcache_read);
}

bool TmaModel::load_row_chunk_packet(uint64_t base_addr,
                                     uint32_t row_stride_bytes,
                                     uint32_t row_idx,
                                     uint32_t chunk_idx,
                                     uint32_t row_bytes,
                                     TmemPacket* out,
                                     const ReadCallback& dcache_read) const {
  if (nullptr == out || !dcache_read) {
    return false;
  }
  out->bytes.fill(0);
  auto byte_offset = chunk_idx * Tmem::kPacketBytes;
  if (byte_offset >= row_bytes) {
    return true;
  }
  auto valid_bytes = std::min<uint32_t>(Tmem::kPacketBytes, row_bytes - byte_offset);
  dcache_read(out->bytes.data(),
              base_addr + static_cast<uint64_t>(row_idx) * row_stride_bytes + byte_offset,
              valid_bytes);
  return true;
}

bool TmaModel::load_payload(const TmaDescriptor& desc,
                            uint32_t capacity,
                            std::vector<uint8_t>* out,
                            const ReadCallback& dcache_read) const {
  // TMA payload traffic is still described in mathematical row-major order at
  // this boundary. Any conversion into TMEM-internal window layout happens
  // later inside the Core/TMEM packet adapter path.
  if (nullptr == out || !dcache_read) {
    return false;
  }

  uint32_t size_bytes = std::min<uint32_t>(payload_size_bytes(desc), capacity);
  out->assign(size_bytes, 0);
  if (size_bytes == 0) {
    return true;
  }

  dcache_read(out->data(), desc.addr, size_bytes);
  return true;
}

bool TmaModel::load_meta(const TmaDescriptor& desc,
                         uint32_t capacity,
                         std::vector<uint8_t>* out,
                         const ReadCallback& dcache_read) const {
  // Sparse metadata is modeled as a separate row-major byte image here. TMEM
  // may later place it into a dedicated shadow window.
  if (nullptr == out || !dcache_read || desc.meta_addr == 0 || desc.meta_size_bytes == 0 || desc.meta_col_span == 0) {
    return false;
  }

  uint32_t size_bytes = std::min<uint32_t>(desc.meta_size_bytes, capacity);
  out->assign(size_bytes, 0);
  if (size_bytes != 0) {
    dcache_read(out->data(), desc.meta_addr, size_bytes);
  }
  return true;
}

void TmaModel::store_payload(const TmaDescriptor& desc,
                             const uint8_t* data,
                             uint32_t size_bytes,
                             const WriteCallback& dcache_write) const {
  // The TMA front-end stores the mathematical window image back to device
  // memory. Any TMEM-specific packet layout has already been decoded before
  // this point.
  if (0 == size_bytes || nullptr == data || !dcache_write) {
    return;
  }
  dcache_write(data, desc.addr, size_bytes);
}
