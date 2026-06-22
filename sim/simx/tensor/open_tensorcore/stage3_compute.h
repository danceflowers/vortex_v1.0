// Stage3: ComputePipeline — backpressure matrix multiply pipeline.
//
// Receives ConvertedTile from Stage2 (OperandFetch), fills internal operand
// SRAMs, and drives the TensorCoreTop 8×8 compute array through all subtile ×
// accum-phase primitives. Completed D results are written back to TMEM via
// TmemWriteReq/Rsp and a CompletedMmaJob is emitted so OpenTensorCore can
// commit the original trace.
//
// Internal flow:
//   FILL    Write A/B/C fp9/fp22 data from ConvertedTile into AMem/BMem/CMem
//           (one tick). DMem is cleared.
//   COMPUTE Issue 8×8 primitives from AMem/BMem/CMem into TensorCoreTop.
//           One primitive per tick when tensorcore is ready; backpressure
//           stalls when the pipeline is full. Retired primitives accumulate
//           into DMem (fp22 add).
//   STORE   Dump DMem subtiles back to external-format packets, pack them in
//           TMEM column-major order, and issue per-packet TMEM writes via
//           TmemWriteReq/Rsp.
//   DONE    Push CompletedMmaJob to Output.

#pragma once

#include <simobject.h>
#include <deque>
#include <memory>
#include <unordered_map>
#include "otc_types.h"
#include "tensor_mem_port_types.h"
#include "open_tensorcore/tensor_compute/tensor_core_top.h"
#include "open_tensorcore/tensor_compute/amem.h"
#include "open_tensorcore/tensor_compute/bmem.h"
#include "open_tensorcore/tensor_compute/cmem.h"
#include "open_tensorcore/tensor_compute/dmem.h"
#include "open_tensorcore/tensor_compute/collector_buffer.h"
#include "tensor_mem_port_types.h"

namespace vortex {

class Core;

struct ComputeConfig {
  uint32_t input_depth       = 2;  // allows next B chunk to queue during COMPUTE
  uint32_t output_depth      = 1;  // Output port (CompletedMmaJob; read via front/pop)
  uint32_t tmem_wrsp_depth   = 2;  // TMEM write response port
  // TmemWriteReq is a cap-0 relay port (no depth needed).
};

class ComputePipeline : public SimObject<ComputePipeline> {
public:
  SimPort<std::shared_ptr<ConvertedTile>> Input;
  SimPort<CompletedMmaJob>                Output;

  SimPort<TensorMemPortReq> TmemWriteReq;
  SimPort<TensorMemPortRsp> TmemWriteRsp;

  ComputePipeline(const SimContext& ctx, const char* name, Core* core,
                  DMem* dmem, const ComputeConfig& config = ComputeConfig{});

  void reset();
  void tick();

  // Collector buffer readiness queries (called by OpenTensorCore::collector_ready).
  bool abuf_ready(uint32_t wid, uint8_t collector_buffer) const {
    return abuf_.is_ready_for(wid, collector_buffer);
  }
  bool bbuf_ready(int8_t bbuf_idx, uint32_t wid, uint8_t collector_buffer) const {
    if (bbuf_idx < 0 || bbuf_idx >= 4) return true;  // no bbuf involved
    return bbuf_[bbuf_idx].is_ready_for(wid, collector_buffer);
  }

private:
  enum class ComputeState : uint8_t { IDLE, FILL, COMPUTE, STORE, DONE };

  static constexpr uint32_t kPrimitiveDim = 8;
  static constexpr uint32_t kSubtiles     = 4;
  static constexpr uint32_t kDenseAccumPhases  = 4;
  static constexpr uint32_t kSparseAccumPhases = 2;

  struct ActiveJob {
    std::shared_ptr<ConvertedTile> tile;
    ComputeState state = ComputeState::IDLE;

    uint32_t issue_subtile       = 0;
    uint32_t issue_accum_phase   = 0;
    uint32_t final_retired_subtiles = 0;

    // Ping-pong BMem state.
    uint32_t active_bmem  = 0;
    uint32_t b_chunk_recv = 0;
    uint32_t total_b_chunks = 1;
    uint32_t round_issued_subtiles = 0;
    uint32_t phases_per_round = 2;

    std::vector<TmemPacket> d_store_packets;
    uint32_t d_packet_idx = 0;
    uint64_t pending_wrsp_id = 0;  // outstanding TMEM write request

    void reset() {
      tile.reset();
      state = ComputeState::IDLE;
      issue_subtile = 0; issue_accum_phase = 0;
      final_retired_subtiles = 0;
      d_store_packets.clear(); d_packet_idx = 0;
      pending_wrsp_id = 0;
      active_bmem = 0; b_chunk_recv = 0; total_b_chunks = 1;
      round_issued_subtiles = 0; phases_per_round = 2;
    }
  };

  void fill_abc(ActiveJob& job);
  void advance_compute(ActiveJob& job);
  void fill_b_bank(uint32_t bank_idx, const std::shared_ptr<ConvertedTile>& tile);
  bool try_receive_next_chunk(ActiveJob& job);
  bool try_swap_to_next_round(ActiveJob& job);
  bool build_store_packets(ActiveJob& job);
  bool advance_store(ActiveJob& job);
  void finish_job(ActiveJob& job);

  void drain_tmem_responses();
  uint64_t issue_tmem_write(uint32_t taddr, uint32_t pkt_idx,
                            const TmemPacket& data);

  static uint32_t accum_phase_count(uint32_t sparsity_kind);
  static uint32_t mem_k_phase(uint32_t sparsity_kind, uint32_t accum_phase);
  static uint32_t fmt_element_bytes(uint32_t fmt);

  Core*         core_;
  ComputeConfig config_;
  DMem*         dmem_;
  AMem          amem_;
  BMem          bmem_[2];
  CMem          cmem_;
  TensorCoreTop tensorcore_;

  // Collector buffers for warp-level ownership tracking.
  ABuf abuf_;
  MBuf mbuf_;
  BBuf bbuf_[4];
  std::deque<ActiveJob> pending_;

  std::array<uint8_t, 64> staged_sparse_meta_;
  bool staged_has_sparse_meta_ = false;

  std::unordered_map<uint64_t, TensorMemPortRsp> completed_tmem_wrsp_;
  uint64_t next_request_id_ = 1;
};

}  // namespace vortex
