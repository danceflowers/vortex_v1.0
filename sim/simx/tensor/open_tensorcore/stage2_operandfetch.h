// Stage2: OperandFetchStage — read A/C from TMEM and B from LMEM, then convert
// to internal precision (A/B → FP9, C → FP22) and package as ConvertedTile.
//
// Receives DecodedMmaJob from Stage1 (TcDecode). Each MMA job is processed
// through a sequential state machine that reads one 64B packet per tick:
//
//   FETCH_A     Issue TMEM read requests for A packets via TmemReadReq/Rsp.
//   FETCH_META  Issue TMEM read for sparse metadata (if sparsity_kind != 0).
//   FETCH_C     Issue TMEM read requests for C/D-input (if enable_input_d != 0).
//   FETCH_B     Read B matrix packets from LMEM via functional lmem_read.
//               TODO: Switch to LmemReadReq/Rsp SimPort path.
//   CONVERT     Precision-convert all raw data in one tick and assemble
//               the ConvertedTile.
//
// TMEM reads share one 512-bit port (TmemReadReq/Rsp, sequenced). Requests
// carry the PTX TADDR + packet index; the TMEM backend resolves the physical
// location.

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
  uint32_t input_depth       = 2;
  uint32_t output_depth      = 1;
  uint32_t tmem_read_depth   = 2;
  uint32_t tmem_rsp_depth    = 2;
  uint32_t lmem_read_depth   = 2;
  uint32_t lmem_rsp_depth    = 2;
  uint32_t convert_latency   = 10;
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
  enum class FetchState : uint8_t {
    IDLE,
    FETCH_A,
    FETCH_META,
    FETCH_C,
    FETCH_B,
    FETCH_B_CHUNK,   // load subsequent B chunk from LMEM (non-first)
    CONVERT,
    CONVERT_B,        // convert B-only chunk and push MID/EOP tile
  };

  enum class TmemReqTag : uint8_t { NONE = 0, A, META, C };

  static constexpr uint32_t kPacketBytes      = 64;
  static constexpr uint32_t kSparseMetaOffset = 512;

  struct PendingFetch {
    DecodedMmaJob job;
    FetchState    state = FetchState::IDLE;

    uint64_t   pending_req_id = 0;
    TmemReqTag pending_tag    = TmemReqTag::NONE;

    // A matrix.
    uint32_t a_packet_count = 0;
    uint32_t a_packet_idx   = 0;
    std::vector<TmemPacket> a_packets;

    // Sparse metadata.
    bool     need_meta = false;
    TmemPacket meta_packet;

    // C / D-input.
    bool     need_c         = false;
    uint32_t c_packet_count = 0;
    uint32_t c_packet_idx   = 0;
    std::vector<TmemPacket> c_packets;

    // B matrix (LMEM).
    uint32_t b_packet_count = 0;
    uint32_t b_packet_idx   = 0;
    uint64_t b_lmem_base    = 0;
    std::vector<std::array<uint8_t, 64>> b_packets;

    // Multi-chunk B streaming for sparse modes.
    uint8_t  total_b_chunks = 1;
    uint8_t  b_chunk_idx    = 0;
    uint64_t b_chunk_base   = 0;
    uint32_t b_chunk_bytes  = 0;

    void reset() {
      state = FetchState::IDLE;
      pending_req_id = 0; pending_tag = TmemReqTag::NONE;
      a_packet_count = 0; a_packet_idx = 0; a_packets.clear();
      need_meta = false; meta_packet = {};
      need_c    = false; c_packet_count = 0; c_packet_idx = 0; c_packets.clear();
      b_packet_count = 0; b_packet_idx = 0; b_lmem_base = 0; b_packets.clear();
      total_b_chunks = 1; b_chunk_idx = 0;
      b_chunk_base = 0; b_chunk_bytes = 0;
    }
  };

  void start_fetch(const DecodedMmaJob& job);
  void drain_tmem_responses();
  void drain_lmem_responses();
  uint64_t issue_tmem_read(uint32_t taddr, uint32_t pkt_idx);
  uint64_t issue_lmem_read(uint64_t addr);
  void advance_fetch(PendingFetch& f);

  bool read_one_a_packet(PendingFetch& f);
  bool read_meta_packet(PendingFetch& f);
  bool read_one_c_packet(PendingFetch& f);
  bool read_one_b_packet(PendingFetch& f);
  bool read_one_b_chunk_packet(PendingFetch& f);
  std::shared_ptr<ConvertedTile> build_b_only_tile(PendingFetch& f);

  std::shared_ptr<ConvertedTile> build_converted_tile(PendingFetch& f);

  Core*               core_;
  OperandFetchConfig  config_;
  std::deque<PendingFetch> pending_;
  std::unordered_map<uint64_t, TensorMemPortRsp> completed_tmem_rsp_;
  std::unordered_map<uint64_t, LsuRsp>       completed_lmem_rsp_;
  uint64_t next_request_id_ = 1;
};

}  // namespace vortex
