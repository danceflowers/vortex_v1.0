// Stage2: OperandFetchStage — read A/C from TMEM and B from LMEM, then convert
// to internal precision (A/B → FP9, C → FP22) and package as ConvertedTile.
//
// Receives DecodedMmaJob from Stage1 (TcDecode). Each MMA job is processed by
// two independent sub-state machines (TMEM and LMEM) that share two separate
// physical ports and can issue/receive requests concurrently:
//
//   TMEM path (TmemReadReq/Rsp, 512-bit port):
//     A     Read A matrix packets from TMEM (4–8 packets, depending on fmt_a)
//     META  Read sparse metadata from TMEM (1 packet, if sparsity_kind != 0)
//     C     Read C/D-input packets from TMEM (0–16 packets, depending on fmt_c)
//     DONE  All TMEM reads complete
//
//   LMEM path (LmemReadReq/Rsp, LSU-style port):
//     B       Read first B chunk from LMEM (4–8 packets, depending on fmt_b)
//     B_CHUNK Read subsequent B chunks (sparse multi-chunk only)
//     DONE    All LMEM reads complete
//
// Send/receive decoupling — each tick:
//   1. RECEIVE: Drain one TMEM and one LMEM response (if available), match to
//      in-flight entries via request_id, store packet data at correct index.
//   2. SEND:    Issue at most one request per port (if port free and more
//      packets remain). TMEM and LMEM requests can issue in the same tick.
//   3. CHECK:   If both TMEM and LMEM phases are DONE and all in-flight
//      requests have been matched, proceed to CONVERT.

#pragma once

#include <simobject.h>
#include <deque>
#include <memory>
#include <unordered_map>
#include "types.h"
#include "otc_types.h"
#include "tensor_mem_port_types.h"

namespace vortex {

class Core;

struct OperandFetchConfig {
  uint32_t input_depth       = 1;
  uint32_t output_depth      = 1;
  uint32_t tmem_read_depth   = 1;
  uint32_t tmem_rsp_depth    = 1;
  uint32_t lmem_read_depth   = 1;
  uint32_t lmem_rsp_depth    = 1;
  uint32_t convert_latency   = 2;
};

class OperandFetchStage : public SimObject<OperandFetchStage> {
public:
  SimPort<DecodedMmaJob> Input;
  SimPort<std::shared_ptr<ConvertedTile>> Output;

  SimPort<TensorMemPortReq> TmemReadReq;
  SimPort<TensorMemPortRsp> TmemReadRsp;

  SimPort<LsuReq> LmemReadReq;
  SimPort<LsuRsp> LmemReadRsp;

  OperandFetchStage(const SimContext& ctx, const char* name, Core* core,
                    const OperandFetchConfig& config = OperandFetchConfig{});

  void reset();
  void tick();

private:
  enum class TmemReqTag : uint8_t { NONE = 0, A, META, C };

  static constexpr uint32_t kPacketBytes      = 64;
  static constexpr uint32_t kSparseMetaOffset = 512;
  static constexpr uint32_t kMaxTmemInflight  = 2;
  static constexpr uint32_t kMaxLmemInflight  = 2;

  // ---------------------------------------------------------------------------
  // PendingFetch — one in-flight MMA operand-fetch job.
  //
  // TMEM and LMEM reads proceed independently via separate sub-state machines,
  // each with its own in-flight request tracker (bounded by port depth).
  //
  // All packets have been received when:
  //   tmem_phase == DONE && tmem_if_count == 0 &&
  //   lmem_phase == DONE && lmem_if_count == 0
  // ---------------------------------------------------------------------------
  struct PendingFetch {
    DecodedMmaJob job;

    // ---- Independent sub-states ----
    enum class TmemPhase : uint8_t { A, META, C, DONE };
    enum class LmemPhase : uint8_t { B, B_CHUNK, DONE };
    TmemPhase tmem_phase = TmemPhase::A;
    LmemPhase lmem_phase = LmemPhase::B;

    // ---- In-flight tracking (per-port, bounded by SimPort depth) ----
    struct TmemInflight {
      uint64_t   req_id = 0;
      TmemReqTag tag    = TmemReqTag::NONE;
      uint32_t   pkt_idx = 0;
    };
    uint8_t      tmem_if_count = 0;
    TmemInflight tmem_if[kMaxTmemInflight];

    struct LmemInflight {
      uint64_t req_id  = 0;
      uint64_t addr    = 0;     // LMEM byte address for functional read on rsp
      uint32_t pkt_idx = 0;     // index into b_packets
    };
    uint8_t      lmem_if_count = 0;
    LmemInflight lmem_if[kMaxLmemInflight];

    // ---- Packet buffers ----
    // A matrix (TMEM).
    uint32_t a_packet_count = 0;
    uint32_t a_packet_idx   = 0;   // next packet to issue (send cursor)
    std::vector<TmemPacket> a_packets;

    // Sparse metadata (TMEM).
    bool       need_meta = false;
    TmemPacket meta_packet;

    // C / D-input (TMEM).
    bool       need_c         = false;
    uint32_t   c_packet_count = 0;
    uint32_t   c_packet_idx   = 0;   // next packet to issue (send cursor)
    std::vector<TmemPacket> c_packets;

    // B matrix (LMEM).
    uint32_t b_packet_count = 0;
    uint32_t b_packet_idx   = 0;   // next packet to issue (send cursor)
    uint64_t b_lmem_base    = 0;
    std::vector<std::array<uint8_t, 64>> b_packets;

    // Multi-chunk B streaming for sparse modes.
    uint8_t  total_b_chunks = 1;
    uint8_t  b_chunk_idx    = 0;
    uint64_t b_chunk_base   = 0;
    uint32_t b_chunk_bytes  = 0;

    // Lifecycle.
    bool completed = false;

    // ---- Queries ----
    bool tmem_phase_done() const {
      return tmem_phase == TmemPhase::DONE && tmem_if_count == 0;
    }
    bool lmem_phase_done() const {
      return lmem_phase == LmemPhase::DONE && lmem_if_count == 0;
    }
    bool all_packets_received() const {
      return tmem_phase_done() && lmem_phase_done();
    }

    void reset() {
      job = {};
      tmem_phase = TmemPhase::A;
      lmem_phase = LmemPhase::B;
      tmem_if_count = 0;
      lmem_if_count = 0;
      a_packet_count = 0; a_packet_idx = 0; a_packets.clear();
      need_meta = false; meta_packet = {};
      need_c    = false; c_packet_count = 0; c_packet_idx = 0; c_packets.clear();
      b_packet_count = 0; b_packet_idx = 0; b_lmem_base = 0; b_packets.clear();
      total_b_chunks = 1; b_chunk_idx = 0;
      b_chunk_base = 0; b_chunk_bytes = 0;
      completed = false;
    }
  };

  void start_fetch(const DecodedMmaJob& job);
  void drain_tmem_responses();
  void drain_lmem_responses();
  uint64_t issue_tmem_read(uint32_t taddr, uint32_t pkt_idx);
  uint64_t issue_lmem_read(uint64_t addr);
  void advance_fetch(PendingFetch& f);

  // Send-phase helpers: try to issue one more request on the given port.
  // Returns true if a request was issued, false if port full or no more packets.
  bool try_issue_next_tmem(PendingFetch& f);
  bool try_issue_next_lmem(PendingFetch& f);

  // Receive-phase: process one matching TMEM/LMEM response.
  void receive_tmem_response(PendingFetch& f);
  void receive_lmem_response(PendingFetch& f);

  // Convert and push.
  void convert_and_push(PendingFetch& f);

  std::shared_ptr<ConvertedTile> build_converted_tile(PendingFetch& f, bool is_sop);

  Core*               core_;
  OperandFetchConfig  config_;
  std::deque<PendingFetch> pending_;
  std::unordered_map<uint64_t, TensorMemPortRsp> completed_tmem_rsp_;
  std::unordered_map<uint64_t, LsuRsp>       completed_lmem_rsp_;
  uint64_t next_request_id_ = 1;
};

}  // namespace vortex
