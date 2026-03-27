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
                header.mma_desc_count * sizeof(MmaDescriptor));
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

bool TmaModel::read_mma_descriptor(uint64_t startup_arg,
                                   uint32_t desc_id,
                                   MmaDescriptor* out,
                                   const ReadCallback& dcache_read) {
  if (nullptr == out || !ensure_descriptor_tables_loaded(startup_arg, dcache_read) || desc_id >= mma_desc_table_.size()) {
    return false;
  }
  *out = mma_desc_table_.at(desc_id);
  return true;
}

uint32_t TmaModel::estimate_load_latency(const TmaDescriptor& desc, bool transpose_b) const {
  uint32_t size_bytes = payload_size_bytes(desc);
  if (desc.meta_addr != 0 && desc.meta_size_bytes != 0 && desc.meta_col_span != 0) {
    size_bytes += desc.meta_size_bytes;
  }
  uint32_t transfer_cycles = std::max<uint32_t>(1, (size_bytes + kLoadBytesPerCycle - 1) / kLoadBytesPerCycle);
  uint32_t latency = kLoadBaseLatency + transfer_cycles;
  if (transpose_b || ((desc.flags & 0x1) != 0)) {
    latency += kTransposePenalty;
  }
  return latency;
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

bool TmaModel::load_payload(const TmaDescriptor& desc,
                            bool transpose_b,
                            uint32_t capacity,
                            std::vector<uint8_t>* out,
                            const ReadCallback& dcache_read) const {
  if (nullptr == out || !dcache_read) {
    return false;
  }

  uint32_t size_bytes = std::min<uint32_t>(payload_size_bytes(desc), capacity);
  out->assign(size_bytes, 0);
  if (size_bytes == 0) {
    return true;
  }

  bool do_transpose = transpose_b || ((desc.flags & 0x1) != 0);
  uint32_t matrix_bytes = desc.rows * desc.cols * desc.elem_bytes;
  if (desc.rows > 0 && desc.cols > 0 && desc.elem_bytes > 0 && do_transpose) {
    std::vector<uint8_t> src(matrix_bytes, 0);
    dcache_read(src.data(), desc.addr, matrix_bytes);
    for (uint32_t r = 0; r < desc.rows; ++r) {
      for (uint32_t c = 0; c < desc.cols; ++c) {
        auto src_off = (r * desc.cols + c) * desc.elem_bytes;
        auto dst_off = (c * desc.rows + r) * desc.elem_bytes;
        if (dst_off + desc.elem_bytes <= out->size() && src_off + desc.elem_bytes <= src.size()) {
          std::copy_n(src.data() + src_off, desc.elem_bytes, out->data() + dst_off);
        }
      }
    }
  } else {
    dcache_read(out->data(), desc.addr, size_bytes);
  }
  return true;
}

bool TmaModel::load_meta(const TmaDescriptor& desc,
                         uint32_t capacity,
                         std::vector<uint8_t>* out,
                         const ReadCallback& dcache_read) const {
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
  if (0 == size_bytes || nullptr == data || !dcache_write) {
    return;
  }
  dcache_write(data, desc.addr, size_bytes);
}
