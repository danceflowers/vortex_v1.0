// Stage2 OperandFetchStage implementation.
//
// TMEM and LMEM reads proceed independently on separate physical ports with
// send/receive decoupling: each tick drains one response per port, then issues
// at most one request per port. TMEM and LMEM can issue simultaneously in the
// same tick, overlapping the two physical interfaces.
//
// In-flight requests are tracked per-port (bounded by SimPort depth) and
// matched to responses via request_id. Packet data is stored at the correct
// buffer index so out-of-order responses are handled correctly.

#include "stage2_operandfetch.h"

#include <cstring>

#include "core.h"
#include "debug.h"
#include "open_tensorcore/tensor_compute/amem.h"
#include "open_tensorcore/tensor_compute/bmem.h"
#include "open_tensorcore/tensor_compute/cmem.h"
#include "tensor_cfg.h"
#include "tensor_mem_port_types.h"

namespace vortex {

namespace vt = vortex::tensor;

namespace {

// Decode b_sdesc → LMEM byte address (PTX §9.7.16.2).
//
// b_sdesc layout (64-bit):
//   [13:0]   start_address  (16 B granularity, relative to shared-mem base)
//   [31:17]  leading_dim     (stride between rows, in 16 B units)
//   [63:62]  swizzle         (0=none, 1=32B, 2=64B, 3=128B)
//
// TODO: swizzle / interleave / leading-dimension are not yet applied.
//       The current path uses a contiguous linear address; full PTX layout
//       modelling will be added when the LMEM backend port is bound and a
//       proper address-calc pipeline is introduced.
uint64_t sdesc_to_lmem_addr(uint64_t sdesc) {
  uint32_t smem_offset = static_cast<uint32_t>(sdesc & 0x3fffu) * 16u;
  return static_cast<uint64_t>(LMEM_BASE_ADDR) + smem_offset;
}

uint32_t tmem_packet_count(uint32_t fmt, bool is_cd) {
  switch (fmt) {
  case vt::fp8::id:  return 4;
  case vt::fp16::id: return 8;
  case vt::fp32::id: return is_cd ? 16 : 0;
  default:            return 0;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Constructor / reset / tick
// ---------------------------------------------------------------------------

OperandFetchStage::OperandFetchStage(const SimContext& ctx, const char* name,
                                     Core* core,
                                     const OperandFetchConfig& config)
  : SimObject<OperandFetchStage>(ctx, name)
  , Input(this, config.input_depth)
  , Output(this, 0)            // source/relay port: must be capacity 0 for bind()
  , TmemReadReq(this, 0)       // source/relay port (→ TMEM read arbiter)
  , TmemReadRsp(this, config.tmem_rsp_depth)
  , LmemReadReq(this, 0)       // source/relay port (→ lmem_arb)
  , LmemReadRsp(this, config.lmem_rsp_depth)
  , core_(core)
  , config_(config) {
  this->reset();
}

void OperandFetchStage::reset() {
  pending_.clear();
  completed_tmem_rsp_.clear();
  completed_lmem_rsp_.clear();
  next_request_id_ = 1;
}

void OperandFetchStage::tick() {
  // Drain one response per port per tick.
  drain_tmem_responses();
  drain_lmem_responses();

  // Accept new job if there is room.
  if (!Input.empty() && pending_.size() < config_.input_depth) {
    auto job = Input.front();
    Input.pop();
    start_fetch(job);
  }

  if (!pending_.empty()) {
    advance_fetch(pending_.front());
    if (pending_.front().completed) {
      pending_.pop_front();
    }
  }
}

// ---------------------------------------------------------------------------
// TMEM helpers
// ---------------------------------------------------------------------------

void OperandFetchStage::drain_tmem_responses() {
  if (!TmemReadRsp.empty()) {
    auto rsp = TmemReadRsp.front();
    TmemReadRsp.pop();
    completed_tmem_rsp_[rsp.request_id] = rsp;
  }
}

uint64_t OperandFetchStage::issue_tmem_read(uint32_t taddr, uint32_t pkt_idx) {
  if (TmemReadReq.full()) return 0;

  TensorMemPortReq req;
  req.request_id            = next_request_id_++;
  req.access_type           = TensorMemPortReq::AccessType::Read;
  req.taddr                 = taddr;
  req.packet_idx            = pkt_idx;

  DT(4, "OperandFetchStage: tmem read req_id=" << req.request_id
     << " taddr=0x" << std::hex << taddr << std::dec
     << " pkt=" << pkt_idx);

  TmemReadReq.push(req, 0);
  return req.request_id;
}

// ---------------------------------------------------------------------------
// LMEM helpers
// ---------------------------------------------------------------------------

void OperandFetchStage::drain_lmem_responses() {
  if (!LmemReadRsp.empty()) {
    auto rsp = LmemReadRsp.front();
    LmemReadRsp.pop();
    completed_lmem_rsp_[rsp.tag] = rsp;
  }
}

uint64_t OperandFetchStage::issue_lmem_read(uint64_t addr) {
  // Issue LsuReq to LMEM for timing (crossbar + bank conflict).
  // Data is read functionally after response (时序功能分离).
  if (LmemReadReq.full()) return 0;
  LsuReq req(LSU_CHANNELS);
  req.write = false;
  req.tag   = static_cast<uint32_t>(next_request_id_++);
  req.addrs.at(0) = addr;
  req.mask.set(0);
  LmemReadReq.push(req, 0);
  return req.tag;
}

// ---------------------------------------------------------------------------
// Job enqueue
// ---------------------------------------------------------------------------

void OperandFetchStage::start_fetch(const DecodedMmaJob& job) {
  PendingFetch f;
  f.job   = job;

  // Determine if collector buffer covers operand reads (USE/LASTUSE).
  uint8_t cb  = job.collector_buffer;
  int8_t  bbi = job.bbuf_idx;
  bool use_abuf_skip = (bbi < 0) && (cb == 0x1 || cb == 0x2);   // ws=0 USE/LASTUSE
  bool use_bbuf_skip = (bbi >= 0) && (cb == 0x1 || cb == 0x2);  // ws=1 USE/LASTUSE

  f.need_c = (job.enable_input_d != 0);
  if (f.need_c) {
    f.c_packet_count = tmem_packet_count(job.fmt_c, true);
    f.c_packets.resize(f.c_packet_count);
  }

  // TMEM path (A/META/C): skip A and META for USE/LASTUSE with ws=0.
  if (use_abuf_skip) {
    f.a_packet_count = 0;
    f.need_meta = false;
    f.tmem_phase = f.need_c ? PendingFetch::TmemPhase::C
                            : PendingFetch::TmemPhase::DONE;
  } else {
    f.a_packet_count = tmem_packet_count(job.fmt_a, false);
    f.a_packets.resize(f.a_packet_count);
    f.need_meta = (job.sparsity_kind != vt::sparse_none);
  }

  // LMEM path (B): skip B entirely for USE/LASTUSE with ws=1.
  if (use_bbuf_skip) {
    f.b_packet_count = 0;
    f.lmem_phase = PendingFetch::LmemPhase::DONE;
  } else {
    f.b_packet_count = tmem_packet_count(job.fmt_b, false);
    f.b_packets.resize(f.b_packet_count);
    f.b_lmem_base = sdesc_to_lmem_addr(job.b_sdesc);
  }

  // Compute B chunk parameters for sparse multi-chunk streaming.
  if (use_bbuf_skip) {
    f.total_b_chunks = 1;
    f.b_chunk_bytes = 0;
    f.b_chunk_base  = 0;
    f.b_chunk_idx   = 0;
  } else {
    uint32_t elem_bytes = 0;
    switch (job.fmt_b) {
    case vt::fp8::id:  elem_bytes = 1; break;
    case vt::fp16::id: elem_bytes = 2; break;
    default:            elem_bytes = 1; break;
    }
    f.b_chunk_bytes = 16 * 16 * elem_bytes;

    f.total_b_chunks = 1;
    if (job.sparsity_kind == vt::sparse_2_4) f.total_b_chunks = 2;
    else if (job.sparsity_kind == vt::sparse_1_4) f.total_b_chunks = 4;
    f.b_chunk_idx  = 0;
    f.b_chunk_base = f.b_lmem_base;
  }

  DT(3, "OperandFetchStage: start wid=" << job.wid
     << " a_taddr=0x" << std::hex << job.a_taddr
     << " d_taddr=0x" << job.d_taddr << std::dec
     << " a_pkts=" << f.a_packet_count
     << " b_pkts=" << f.b_packet_count
     << " c_pkts=" << f.c_packet_count
     << " b_chunks=" << (int)f.total_b_chunks);

  pending_.push_back(std::move(f));
}

// ---------------------------------------------------------------------------
// State machine — send/receive decoupled
//
// Each tick:
//   1. RECEIVE: Match ALL completed responses to in-flight entries.
//   2. SEND:    Issue as many new requests as port free slots allow.
//   3. CHECK:   If all packets received, convert and push tile.
// ---------------------------------------------------------------------------

void OperandFetchStage::advance_fetch(PendingFetch& f) {
  // =========================================================================
  // 1. RECEIVE PHASE: drain all completed responses
  // =========================================================================

  // --- TMEM responses ---
  receive_tmem_response(f);

  // --- LMEM responses ---
  receive_lmem_response(f);

  // =========================================================================
  // 2. SEND PHASE: issue one request per port per tick
  // =========================================================================

  // --- TMEM ---
  if (f.tmem_if_count < kMaxTmemInflight
      && !TmemReadReq.full()
      && f.tmem_phase != PendingFetch::TmemPhase::DONE) {
    try_issue_next_tmem(f);
  }

  // --- LMEM ---
  if (f.lmem_if_count < kMaxLmemInflight
      && !LmemReadReq.full()
      && f.lmem_phase != PendingFetch::LmemPhase::DONE) {
    try_issue_next_lmem(f);
  }

  // =========================================================================
  // 3. CHECK: all packets received → CONVERT
  // =========================================================================
  if (f.all_packets_received()) {
    convert_and_push(f);
  }
}

// ---------------------------------------------------------------------------
// Receive helpers — match responses to in-flight entries by request_id
// ---------------------------------------------------------------------------

void OperandFetchStage::receive_tmem_response(PendingFetch& f) {
  for (uint8_t i = 0; i < f.tmem_if_count; ) {
    auto it = completed_tmem_rsp_.find(f.tmem_if[i].req_id);
    if (it == completed_tmem_rsp_.end()) { ++i; continue; }

    auto& tif = f.tmem_if[i];
    switch (tif.tag) {
    case TmemReqTag::A:
      f.a_packets.at(tif.pkt_idx).bytes = it->second.read_packet.bytes;
      break;
    case TmemReqTag::META:
      f.meta_packet.bytes = it->second.read_packet.bytes;
      break;
    case TmemReqTag::C:
      f.c_packets.at(tif.pkt_idx).bytes = it->second.read_packet.bytes;
      break;
    default: break;
    }
    completed_tmem_rsp_.erase(it);

    // Remove from in-flight: swap with last, decrement count.
    f.tmem_if[i] = f.tmem_if[--f.tmem_if_count];
    // Don't increment i — the swapped-in entry at position i may also match.
  }
}

void OperandFetchStage::receive_lmem_response(PendingFetch& f) {
  for (uint8_t i = 0; i < f.lmem_if_count; ) {
    auto it = completed_lmem_rsp_.find(f.lmem_if[i].req_id);
    if (it == completed_lmem_rsp_.end()) { ++i; continue; }

    auto& lif = f.lmem_if[i];
    // Functional read: the LsuReq provided timing; now get the actual data.
    core_->lmem_read(f.b_packets.at(lif.pkt_idx).data(), lif.addr, kPacketBytes);
    completed_lmem_rsp_.erase(it);

    // Remove from in-flight.
    f.lmem_if[i] = f.lmem_if[--f.lmem_if_count];
  }
}

// ---------------------------------------------------------------------------
// Send helpers — try to issue one more request on TMEM / LMEM port
// ---------------------------------------------------------------------------

bool OperandFetchStage::try_issue_next_tmem(PendingFetch& f) {
  switch (f.tmem_phase) {
  case PendingFetch::TmemPhase::A: {
    if (f.a_packet_idx >= f.a_packet_count) {
      // All A packets issued → advance phase.
      f.tmem_phase = f.need_meta ? PendingFetch::TmemPhase::META
                   : f.need_c   ? PendingFetch::TmemPhase::C
                   :              PendingFetch::TmemPhase::DONE;
      return (f.tmem_phase != PendingFetch::TmemPhase::DONE);  // try next phase
    }
    uint64_t req_id = issue_tmem_read(f.job.a_taddr, f.a_packet_idx);
    if (req_id == 0) return false;  // port full
    f.tmem_if[f.tmem_if_count++] = {req_id, TmemReqTag::A, f.a_packet_idx};
    ++f.a_packet_idx;
    return true;
  }

  case PendingFetch::TmemPhase::META: {
    uint32_t meta_pkt = (kSparseMetaOffset
                        + f.job.sparsity_meta_sel * kPacketBytes)
                        / kPacketBytes;
    uint64_t req_id = issue_tmem_read(f.job.a_taddr, meta_pkt);
    if (req_id == 0) return false;
    f.tmem_if[f.tmem_if_count++] = {req_id, TmemReqTag::META, 0};
    f.tmem_phase = f.need_c ? PendingFetch::TmemPhase::C
                            : PendingFetch::TmemPhase::DONE;
    return true;
  }

  case PendingFetch::TmemPhase::C: {
    if (f.c_packet_idx >= f.c_packet_count) {
      f.tmem_phase = PendingFetch::TmemPhase::DONE;
      return false;
    }
    uint64_t req_id = issue_tmem_read(f.job.d_taddr, f.c_packet_idx);
    if (req_id == 0) return false;
    f.tmem_if[f.tmem_if_count++] = {req_id, TmemReqTag::C, f.c_packet_idx};
    ++f.c_packet_idx;
    return true;
  }

  case PendingFetch::TmemPhase::DONE:
  default:
    return false;
  }
}

bool OperandFetchStage::try_issue_next_lmem(PendingFetch& f) {
  switch (f.lmem_phase) {
  case PendingFetch::LmemPhase::B: {
    if (f.b_packet_idx >= f.b_packet_count) {
      f.lmem_phase = PendingFetch::LmemPhase::DONE;
      return false;
    }
    uint64_t addr = f.b_lmem_base
                  + uint64_t(f.b_packet_idx) * kPacketBytes;
    uint64_t req_id = issue_lmem_read(addr);
    if (req_id == 0) return false;
    f.lmem_if[f.lmem_if_count++] = {req_id, addr, f.b_packet_idx};
    ++f.b_packet_idx;
    return true;
  }

  case PendingFetch::LmemPhase::B_CHUNK: {
    if (f.b_packet_idx >= f.b_packet_count) {
      f.lmem_phase = PendingFetch::LmemPhase::DONE;
      return false;
    }
    uint64_t addr = f.b_chunk_base
                  + uint64_t(f.b_chunk_idx) * f.b_chunk_bytes
                  + uint64_t(f.b_packet_idx) * kPacketBytes;
    uint64_t req_id = issue_lmem_read(addr);
    if (req_id == 0) return false;
    f.lmem_if[f.lmem_if_count++] = {req_id, addr, f.b_packet_idx};
    ++f.b_packet_idx;
    return true;
  }

  case PendingFetch::LmemPhase::DONE:
  default:
    return false;
  }
}

// ---------------------------------------------------------------------------
// CONVERT — all packets received, assemble the ConvertedTile
// ---------------------------------------------------------------------------

void OperandFetchStage::convert_and_push(PendingFetch& f) {
  bool is_sop = (f.b_chunk_idx == 0);
  auto tile = build_converted_tile(f, is_sop);
  if (!tile) return;

  DT(3, "OperandFetchStage: " << (is_sop ? "SOP" : "B") << " tile"
     << " chunk=" << (int)f.b_chunk_idx << "/" << (int)f.total_b_chunks
     << " wid=" << f.job.wid);

  Output.push(std::move(tile), config_.convert_latency);

  // Advance to next B chunk or mark complete.
  if (f.b_chunk_idx + 1 < f.total_b_chunks) {
    ++f.b_chunk_idx;
    f.b_packet_idx   = 0;
    f.b_packet_count = tmem_packet_count(f.job.fmt_b, false);
    f.b_packets.clear();
    f.b_packets.resize(f.b_packet_count);
    // Reset LMEM sub-state for next chunk; TMEM stays DONE.
    f.lmem_phase     = PendingFetch::LmemPhase::B_CHUNK;
    f.lmem_if_count  = 0;
    // TMEM stays DONE (no A/C/meta re-read for subsequent chunks).
  } else {
    f.completed = true;
  }
}

// ---------------------------------------------------------------------------
// Precision conversion (unchanged logic)
// ---------------------------------------------------------------------------

std::shared_ptr<ConvertedTile> OperandFetchStage::build_converted_tile(
    PendingFetch& f, bool is_sop) {
  auto tile = std::make_shared<ConvertedTile>();

  // Identity fields (always populated).
  tile->uuid          = f.job.uuid;
  tile->wid           = f.job.wid;
  tile->fmt_a         = f.job.fmt_a;
  tile->fmt_b         = f.job.fmt_b;
  tile->fmt_c         = f.job.fmt_c;
  tile->fmt_d         = f.job.fmt_d;
  tile->shape_m       = f.job.shape_m;
  tile->shape_n       = f.job.shape_n;
  tile->sparsity_kind = f.job.sparsity_kind;
  tile->d_taddr       = f.job.d_taddr;

  // Collector buffer passthrough for Stage3 reuse.
  tile->collector_buffer = f.job.collector_buffer;
  tile->bbuf_idx         = f.job.bbuf_idx;

  // Stream control.
  tile->sop = is_sop ? 1 : 0;
  tile->eop = (f.b_chunk_idx + 1 == f.total_b_chunks) ? 1 : 0;
  tile->b_chunk_idx  = f.b_chunk_idx;
  tile->total_b_chunks = f.total_b_chunks;

  uint8_t  cb  = f.job.collector_buffer;
  int8_t   bbi = f.job.bbuf_idx;
  bool use_abuf = (bbi < 0) && (cb != 0x3);          // ws=0, not DISCARD
  bool use_bbuf = (bbi >= 0) && (cb != 0x3);          // ws=1, not DISCARD

  if (is_sop) {
    // A: 4 lines, each 8×8 → FP9 (only in SOP tile).
    // Skip when USE/LASTUSE reads A from abuf — packets are empty.
    bool skip_a = use_abuf && (cb == 0x1 || cb == 0x2);
    if (!skip_a) {
      uint32_t ppl = AMem::packets_per_fill_line(f.job.fmt_a);
      for (uint32_t line = 0; line < AMem::kDepth; ++line) {
        std::vector<AMem::packet_t> line_pkts;
        line_pkts.reserve(ppl);
        for (uint32_t p = 0; p < ppl; ++p) {
          uint32_t idx = line * ppl + p;
          if (idx >= f.a_packets.size()) return nullptr;
          line_pkts.push_back(f.a_packets.at(idx).bytes);
        }
        uint16_t converted[8][8];
        if (!AMem::convert_fill_packets(f.job.fmt_a, line_pkts, converted))
          return nullptr;
        uint32_t k_phase = line / 2, m_block = line % 2;
        for (uint32_t i = 0; i < 8; ++i)
          for (uint32_t k = 0; k < 8; ++k)
            tile->a_fp9[m_block * 8 + i][k_phase * 8 + k] = converted[i][k];
      }
    }

    // C: 4 subtiles → FP22 (only in SOP tile).
    if (f.need_c) {
      tile->has_c = true;
      uint32_t pps = CMem::packets_per_subtile(f.job.fmt_c);
      for (uint32_t st = 0; st < CMem::kDepth; ++st) {
        std::vector<CMem::packet_t> st_pkts;
        st_pkts.reserve(pps);
        for (uint32_t p = 0; p < pps; ++p) {
          uint32_t idx = st * pps + p;
          if (idx >= f.c_packets.size()) return nullptr;
          st_pkts.push_back(f.c_packets.at(idx).bytes);
        }
        uint32_t converted[8][8];
        if (!CMem::convert_fill_packets(f.job.fmt_c, st_pkts, converted))
          return nullptr;
        uint32_t m_block = st / 2, n_block = st % 2;
        for (uint32_t i = 0; i < 8; ++i)
          for (uint32_t j = 0; j < 8; ++j)
            tile->c_fp22[m_block * 8 + i][n_block * 8 + j] = converted[i][j];
      }
    }

    if (f.need_meta) {
      tile->has_sparse_meta = true;
      tile->sparse_meta = f.meta_packet.bytes;
    }
  }

  // B: 4 lines → FP9 (every tile carries B data).
  // Skip when USE/LASTUSE reads B from bbuf — packets are empty.
  bool skip_b = use_bbuf && (cb == 0x1 || cb == 0x2);
  if (!skip_b) {
    uint32_t ppl = BMem::packets_per_fill_line(f.job.fmt_b);
    for (uint32_t line = 0; line < BMem::kDepth; ++line) {
      std::vector<BMem::packet_t> line_pkts;
      line_pkts.reserve(ppl);
      for (uint32_t p = 0; p < ppl; ++p) {
        uint32_t idx = line * ppl + p;
        if (idx >= f.b_packets.size()) return nullptr;
        line_pkts.push_back(f.b_packets.at(idx));
      }
      uint16_t converted[8][8];
      if (!BMem::convert_fill_packets(f.job.fmt_b, line_pkts, converted))
        return nullptr;
      uint32_t k_phase = line / 2, n_block = line % 2;
      for (uint32_t k = 0; k < 8; ++k)
        for (uint32_t j = 0; j < 8; ++j)
          tile->b_fp9[k_phase * 8 + k][n_block * 8 + j] = converted[k][j];
    }
  }

  return tile;
}

}  // namespace vortex
