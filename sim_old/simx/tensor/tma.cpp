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
#include <stdexcept>
#include <ostream>
#include "tma.h"
#include "core.h"
#include "idescriptor.h"
#include "tmem.h"
#include "VX_config.h"

using namespace vortex;

namespace {

// Clamp invalid tensor-map ranks into the supported 1D..5D range.
uint32_t tensor_map_rank(const tensor_map_t& tmap) {
  return tmap.rank == 0 ? 1u : (tmap.rank > 5 ? 5u : tmap.rank);
}

// Compute the contiguous byte span described by the tensor map box.
uint32_t tensor_map_total_bytes(const tensor_map_t& tmap) {
  auto rank = tensor_map_rank(tmap);
  auto elem_bytes = Tma::element_type_bytes(tmap.element_type);
  if (elem_bytes == 0) {
    elem_bytes = 1;
  }

  uint64_t total_elems = 1;
  for (uint32_t d = 0; d < rank; ++d) {
    auto box = tmap.box_size[d];
    if (box == 0) {
      box = 1;
    }
    total_elems *= box;
  }
  return static_cast<uint32_t>(total_elems * elem_bytes);
}

// Resolve the global byte address for the requested tensor coordinates.
uint64_t tensor_map_address(const tensor_map_t& tmap,
                            const cpabulk_transfer_args_t& args) {
  auto rank = tensor_map_rank(tmap);
  auto elem_bytes = Tma::element_type_bytes(tmap.element_type);
  if (elem_bytes == 0) {
    elem_bytes = 1;
  }

  auto address = tmap.global_address;
  for (uint32_t d = 0; d < rank; ++d) {
    auto stride = tmap.global_stride[d];
    if (stride == 0) {
      stride = (d == 0) ? elem_bytes : tmap.box_size[d - 1] * elem_bytes;
    }
    address += static_cast<uint64_t>(args.coords[d]) * stride;
  }
  return address;
}

} // namespace

Tma::Tma(const SimContext& ctx, const char* name, Core* core)
  : SimObject(ctx, name)
  , TmemReqOut(this)
  , TmemRspIn(this)
  , AsyncOpCompletionOut(this)
  , CacheReqOut(1, this)
  , CacheRspIn(1, this)
  , core_(core)
  , next_lmem_to_tmem_request_id_(1) {
  (void)name;
}

void Tma::reset() {
  pending_lmem_to_tmem_copies_.clear();
  completed_lmem_to_tmem_responses_.clear();
  next_lmem_to_tmem_request_id_ = 1;
}

void Tma::tick() {
  drain_tmem_responses();
  advance_lmem_to_tmem_copy_ops();
}

bool Tma::is_lmem_addr(uint64_t addr) {
  if (addr < LMEM_BASE_ADDR) {
    return false;
  }
  return (addr - LMEM_BASE_ADDR) < (uint64_t{1} << LMEM_LOG_SIZE);
}

uint32_t Tma::element_type_bytes(uint8_t element_type) {
  // CUtensorMapDataType encoding (PTX §6.4.10.4):
  //   0: U8   1: U16  2: U32  3: I32   4: U64
  //   5: I64  6: F16  7: F32  8: F64
  //   9: BF16 10: FP8 11: FP4 ...
  switch (element_type) {
    case 0:  return 1;
    case 1:  return 2;
    case 2:  return 4;
    case 3:  return 4;
    case 4:  return 8;
    case 5:  return 8;
    case 6:  return 2;
    case 7:  return 4;
    case 8:  return 8;
    case 9:  return 2;
    case 10: return 1;
    case 11: return 1;  // 4-bit packed; keep the previous Cmodel behavior.
    default: return 4;
  }
}

cpabulk_transfer_result_t Tma::cpabulk_tensor_load(uint64_t tensor_map_addr,
                                                   uint64_t args_lmem_ptr,
                                                   bool complete_tx) {
  if (!is_lmem_addr(args_lmem_ptr)) {
    throw std::runtime_error(
      "cp.async.bulk.tensor.load requires args_lmem_ptr to point to LMEM; "
      "raw window operands are not part of the PTX path");
  }

  cpabulk_transfer_args_t args = {};
  core_->lmem_read(&args, args_lmem_ptr, sizeof(args));

  tensor_map_t tmap = {};
  core_->dcache_read(&tmap, tensor_map_addr, sizeof(tmap));

  auto total_bytes = tensor_map_total_bytes(tmap);
  auto src_addr = tensor_map_address(tmap, args);

  constexpr uint32_t kChunk = 64;
  uint8_t buf[kChunk];
  auto lmem_dst = static_cast<uint64_t>(args.smem_addr);
  uint32_t copied = 0;
  while (copied < total_bytes) {
    auto take = std::min(kChunk, total_bytes - copied);
    core_->dcache_read(buf, src_addr + copied, take);
    core_->lmem_write(buf, lmem_dst + copied, take);
    copied += take;
  }

  cpabulk_transfer_result_t result{};
  result.payload_size_bytes = total_bytes;
  if (complete_tx && args.mbar_addr != 0) {
    result.tx_bound_mbar = static_cast<uint64_t>(args.mbar_addr);
    result.tx_bytes = total_bytes;
  }
  return result;
}

cpabulk_transfer_result_t Tma::cpabulk_tensor_store(uint64_t tensor_map_addr,
                                                    uint64_t args_lmem_ptr) {
  if (!is_lmem_addr(args_lmem_ptr)) {
    throw std::runtime_error(
      "cp.async.bulk.tensor.store requires args_lmem_ptr to point to LMEM; "
      "raw window operands are not part of the PTX path");
  }

  cpabulk_transfer_args_t args = {};
  core_->lmem_read(&args, args_lmem_ptr, sizeof(args));

  tensor_map_t tmap = {};
  core_->dcache_read(&tmap, tensor_map_addr, sizeof(tmap));

  auto total_bytes = tensor_map_total_bytes(tmap);
  auto dst_addr = tensor_map_address(tmap, args);

  constexpr uint32_t kChunk = 64;
  uint8_t buf[kChunk];
  auto lmem_src = static_cast<uint64_t>(args.smem_addr);
  uint32_t copied = 0;
  while (copied < total_bytes) {
    auto take = std::min(kChunk, total_bytes - copied);
    core_->lmem_read(buf, lmem_src + copied, take);
    core_->dcache_write(buf, dst_addr + copied, take);
    copied += take;
  }

  cpabulk_transfer_result_t result{};
  result.payload_size_bytes = total_bytes;
  return result;
}

bool Tma::issue_lmem_to_tmem_copy(uint32_t async_id,
                                  uint32_t wid,
                                  uint32_t wgid,
                                  uint32_t col_base,
                                  uint32_t col_span,
                                  uint64_t lmem_addr,
                                  uint32_t byte_offset,
                                  uint32_t total_bytes) {
  if (total_bytes == 0) {
    return false;
  }

  pending_lmem_to_tmem_copy_t op{};
  op.async_id = async_id;
  op.wid = wid;
  op.wgid = wgid;
  op.col_base = col_base;
  op.col_span = col_span;
  op.lmem_addr = lmem_addr;
  op.byte_offset = byte_offset;
  op.total_bytes = total_bytes;
  pending_lmem_to_tmem_copies_[async_id] = op;
  return true;
}

void Tma::drain_tmem_responses() {
  if (TmemRspIn.empty()) {
    return;
  }
  auto response = TmemRspIn.front();
  completed_lmem_to_tmem_responses_[response.request_id] = response;
  TmemRspIn.pop();
}

void Tma::advance_lmem_to_tmem_copy_ops() {
  if (pending_lmem_to_tmem_copies_.empty()) {
    return;
  }

  auto issue_request = [this](pending_lmem_to_tmem_copy_t& op,
                              TensorMemPortReq::AccessType access_type) {
    TensorMemPortReq request{};
    request.request_id = next_lmem_to_tmem_request_id_++;
    request.arbitration_age =
        (uint64_t(op.async_id) << 32)
      | (uint64_t(op.packet_idx) << 1)
      | (access_type == TensorMemPortReq::AccessType::Write ? 1ull : 0ull);
    request.access_type = access_type;
    request.port_request.kind =
        (access_type == TensorMemPortReq::AccessType::Write)
      ? Tmem::PortRequestKind::RegionWrite
      : Tmem::PortRequestKind::RegionRead;
    request.port_request.col_base = op.col_base;
    request.port_request.col_span = op.col_span;
    request.port_request.packet_idx = op.packet_idx;
    if (access_type == TensorMemPortReq::AccessType::Write) {
      request.write_packet = op.packet;
    }
    TmemReqOut.push(request, 0);
    op.request_id = request.request_id;
  };

  auto prepare_packet_window = [](pending_lmem_to_tmem_copy_t& op) {
    auto absolute_offset = op.byte_offset + op.cursor;
    op.packet_idx = absolute_offset / Tmem::kPacketBytes;
    op.packet_offset = absolute_offset % Tmem::kPacketBytes;
    op.packet_bytes = std::min<uint32_t>(Tmem::kPacketBytes - op.packet_offset,
                                         op.total_bytes - op.cursor);
  };

  std::vector<uint32_t> completed_ops;
  for (auto& entry : pending_lmem_to_tmem_copies_) {
    auto& op = entry.second;

    if (op.request_id != 0) {
      auto response_it = completed_lmem_to_tmem_responses_.find(op.request_id);
      if (response_it == completed_lmem_to_tmem_responses_.end()) {
        if (op.stage == lmem_to_tmem_copy_stage_t::WaitRead) {
          ++core_->mutable_perf_stats().stall_tmem_read_port_busy;
        } else if (op.stage == lmem_to_tmem_copy_stage_t::WaitWrite) {
          ++core_->mutable_perf_stats().stall_tmem_write_port_busy;
        }
        continue;
      }

      auto response = response_it->second;
      completed_lmem_to_tmem_responses_.erase(response_it);
      op.request_id = 0;

      if (op.stage == lmem_to_tmem_copy_stage_t::WaitRead) {
        ++core_->mutable_perf_stats().tmem_read_packets;
        op.packet = response.read_packet;
        core_->lmem_read(op.packet.bytes.data() + op.packet_offset,
                         op.lmem_addr + op.cursor,
                         op.packet_bytes);
        op.stage = lmem_to_tmem_copy_stage_t::Ready;
        issue_request(op, TensorMemPortReq::AccessType::Write);
        op.stage = lmem_to_tmem_copy_stage_t::WaitWrite;
        continue;
      } else if (op.stage == lmem_to_tmem_copy_stage_t::WaitWrite) {
        ++core_->mutable_perf_stats().tmem_write_packets;
        op.cursor += op.packet_bytes;
        op.stage = lmem_to_tmem_copy_stage_t::Ready;
      }
    }

    if (op.cursor >= op.total_bytes) {
      core_->tmem_set_payload_ready(op.col_base, true);
      AsyncOpCompletionOut.push({op.async_id}, 0);
      completed_ops.push_back(entry.first);
      continue;
    }

    if (op.stage != lmem_to_tmem_copy_stage_t::Ready || op.request_id != 0) {
      continue;
    }

    prepare_packet_window(op);
    if (op.packet_offset == 0 && op.packet_bytes == Tmem::kPacketBytes) {
      core_->lmem_read(op.packet.bytes.data(),
                       op.lmem_addr + op.cursor,
                       Tmem::kPacketBytes);
      issue_request(op, TensorMemPortReq::AccessType::Write);
      op.stage = lmem_to_tmem_copy_stage_t::WaitWrite;
    } else {
      issue_request(op, TensorMemPortReq::AccessType::Read);
      op.stage = lmem_to_tmem_copy_stage_t::WaitRead;
    }
  }

  for (auto async_id : completed_ops) {
    pending_lmem_to_tmem_copies_.erase(async_id);
  }
}

void Tma::dump_debug_state(std::ostream& os) const {
  os << "[Tma] lmem_to_tmem_pending=" << pending_lmem_to_tmem_copies_.size()
     << " completed_tmem_responses=" << completed_lmem_to_tmem_responses_.size()
     << "\n";
  for (const auto& entry : pending_lmem_to_tmem_copies_) {
    const auto& op = entry.second;
    os << "  async_id=" << op.async_id
       << " wid=" << op.wid
       << " wgid=" << op.wgid
       << " col_base=" << op.col_base
       << " col_span=" << op.col_span
       << " cursor=" << op.cursor
       << " total=" << op.total_bytes
       << " stage=" << static_cast<uint32_t>(op.stage)
       << " pending_req=" << op.request_id
       << "\n";
  }
}
