// TMA — Tensor Memory Accelerator (cp.async.bulk.tensor load/store).
//
// Receives instr_trace_t* (with TmaTraceData) from TmaUnit. State machine:
//   DECODE            → extract fields from TmaTraceData
//   READ_TRANSFER_ARGS → read 32B cpabulk_transfer_args_t from LMEM (LmemDescReq/Rsp)
//   READ_TENSOR_MAP   → two 64B MemReq reads from TCache (CacheReq/Rsp)
//   XFER_DATA (LD)    → per-packet: CacheReq(read) → CacheRsp → TmemWriteReq → TmemWriteRsp
//   XFER_DATA (ST)    → per-packet: TmemReadReq → TmemReadRsp → CacheReq(write) → CacheRsp
//   COMPLETE          → AsyncCompletionOut + push trace to Outputs

#pragma once

#include <simobject.h>
#include <deque>
#include <memory>
#include <unordered_map>
#include "types.h"
#include "instr_trace.h"
#include "tensor_mem_port_types.h"
#include "idescriptor.h"

namespace vortex {

class Core;

namespace tensor {

struct TmaConfig {
  uint32_t max_inflight = 4;
};

class Tma : public SimObject<Tma> {
public:
  std::vector<SimPort<instr_trace_t*>> Inputs;
  std::vector<SimPort<instr_trace_t*>> Outputs;

  SimPort<MemReq> CacheReq;
  SimPort<MemRsp> CacheRsp;

  SimPort<TensorMemPortReq> TmemReadReq;
  SimPort<TensorMemPortRsp> TmemReadRsp;
  SimPort<TensorMemPortReq> TmemWriteReq;
  SimPort<TensorMemPortRsp> TmemWriteRsp;

  SimPort<LsuReq> LmemDescReq;
  SimPort<LsuRsp> LmemDescRsp;

  SimPort<TensorAsyncOpCompletion> AsyncCompletionOut;

  Tma(const SimContext& ctx, const char* name, Core* core,
      const TmaConfig& config = TmaConfig{});

  void reset();
  void tick();

private:
  enum class TmaStage : uint8_t {
    IDLE,
    READ_TRANSFER_ARGS,
    READ_TENSOR_MAP_LO,   // first 64B of tensor_map_t
    READ_TENSOR_MAP_HI,   // second 64B
    XFER_DATA,
    COMPLETE,
  };

  enum class ReqTag : uint8_t { NONE, DESC, TMAP_LO, TMAP_HI, XFER_CACHE, XFER_TMEM };

  struct ActiveJob {
    TmaStage  stage = TmaStage::IDLE;
    instr_trace_t* trace = nullptr;
    uint32_t  lane = 0;

    bool      is_load = true;
    // Destination of a load (resolved from smem_addr once the transfer args are
    // read): a TMEM taddr (A/D, written via TmemWriteReq) vs. a SHARED-mem/LMEM
    // address (B, written via lmem_arb). smem_addr < LMEM_BASE_ADDR ⇒ TMEM.
    bool      dest_is_tmem = true;
    uint64_t  tensor_map_addr = 0;    // DRAM addr of tensor_map_t (128 B)
    uint64_t  args_lmem_ptr   = 0;    // LMEM addr of cpabulk_transfer_args_t
    uint32_t  wid = 0;
    uint32_t  async_id = 0;
    bool      complete_tx = false;

    // Parsed tensor_map_t.
    uint8_t   tmap_buf[128] = {};     // raw 128 B
    uint64_t  global_addr = 0;        // tensor base DRAM address
    uint32_t  elem_bytes  = 0;        // bytes per element
    uint32_t  tile_bytes  = 0;        // total bytes in the tile

    // Helper: compute DRAM byte address for linear element index within tile.
    uint64_t compute_elem_addr(uint32_t linear_idx) const {
      // De-linearize to per-dimension coords, then apply global_stride. Per
      // CUtensorMap, global_stride is already in BYTES, so it must not be scaled
      // again by elem_bytes.
      const auto* tmap = reinterpret_cast<const tensor_map_t*>(tmap_buf);
      uint64_t off = 0;
      uint32_t rem = linear_idx;
      for (int d = (int)tmap->rank - 1; d >= 0; --d) {
        uint32_t dim = static_cast<uint32_t>(tmap->box_size[d]);
        uint32_t coord = (dim > 1) ? (rem % dim) : 0;
        rem /= dim;
        off += coord * tmap->global_stride[d];
      }
      return global_addr + off;
    }

    // Parsed cpabulk_transfer_args_t.
    uint32_t  smem_addr = 0;
    uint32_t  mbar_addr = 0;
    int32_t   coords[5] = {};
    uint32_t  taddr = 0;             // TMEM location

    // Per-packet transfer cursor.
    uint32_t  total_pkts = 0;
    uint32_t  next_pkt   = 0;

    uint64_t  pending_req_id = 0;
    ReqTag    pending_tag = ReqTag::NONE;

    // Staging buffer for the current 64B packet while crossing the cache↔TMEM boundary.
    TmemPacket staging_pkt;
  };

  void decode_trace();
  void advance_job(ActiveJob& job);
  void complete_job(ActiveJob& job);

  Core*     core_;

  TmaConfig config_;
  std::deque<ActiveJob> active_;

  // Response maps (request_id → payload).
  std::unordered_map<uint64_t, MemRsp>              cache_rsps_;
  std::unordered_map<uint64_t, LsuRsp>         lmem_rsps_;
  std::unordered_map<uint64_t, TensorMemPortRsp>    tmem_rsps_;
  uint64_t next_req_id_ = 1;
};

}  // namespace tensor
}  // namespace vortex
