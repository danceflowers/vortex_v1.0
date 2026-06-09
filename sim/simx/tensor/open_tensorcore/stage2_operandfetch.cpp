// Stage2 OperandFetchStage implementation.
//
// TMEM reads: push TensorMemPortReq{taddr, packet_idx} to TmemReadReq.
// The TMEM backend (TmemSystem) resolves taddr → physical location internally.
// LMEM reads: functional lmem_read (TODO: switch to LmemReadReq/Rsp).

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
  , Output(this, config.output_depth)
  , TmemReadReq(this, config.tmem_read_depth)
  , TmemReadRsp(this, config.tmem_rsp_depth)
  , LmemReadReq(this, config.lmem_read_depth)
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
  drain_tmem_responses();
  drain_lmem_responses();

  if (!Input.empty() && pending_.size() < config_.input_depth) {
    auto job = Input.front();
    Input.pop();
    start_fetch(job);
  }

  if (!pending_.empty()) {
    advance_fetch(pending_.front());
    if (pending_.front().state == FetchState::IDLE) {
      pending_.pop_front();
    }
  }
}

// ---------------------------------------------------------------------------
// TMEM helpers
// ---------------------------------------------------------------------------

void OperandFetchStage::drain_tmem_responses() {
  while (!TmemReadRsp.empty()) {
    auto rsp = TmemReadRsp.front();
    TmemReadRsp.pop();
    completed_tmem_rsp_[rsp.request_id] = rsp;
  }
}

uint64_t OperandFetchStage::issue_tmem_read(uint32_t taddr, uint32_t pkt_idx) {
  if (TmemReadReq.full()) return 0;

  TensorMemPortReq req;
  req.request_id           = next_request_id_++;
  req.access_type           = TensorMemPortReq::AccessType::Read;
  req.taddr    = taddr;
  req.packet_idx = pkt_idx;

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
  while (!LmemReadRsp.empty()) {
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
  f.state = FetchState::FETCH_A;

  f.a_packet_count = tmem_packet_count(job.fmt_a, false);
  f.a_packets.resize(f.a_packet_count);

  f.need_meta = (job.sparsity_kind != vt::sparse_none);
  f.need_c    = (job.enable_input_d != 0);
  if (f.need_c) {
    f.c_packet_count = tmem_packet_count(job.fmt_c, true);
    f.c_packets.resize(f.c_packet_count);
  }

  f.b_packet_count = tmem_packet_count(job.fmt_b, false);
  f.b_packets.resize(f.b_packet_count);
  f.b_lmem_base = sdesc_to_lmem_addr(job.b_sdesc);

  // Compute B chunk parameters for sparse multi-chunk streaming.
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
// State machine
// ---------------------------------------------------------------------------

void OperandFetchStage::advance_fetch(PendingFetch& f) {
  // Check pending request (TMEM or LMEM depending on stage).
  if (f.pending_req_id != 0) {
    if (f.state <= FetchState::FETCH_C) {
      auto it = completed_tmem_rsp_.find(f.pending_req_id);
      if (it == completed_tmem_rsp_.end()) return;
      switch (f.pending_tag) {
      case TmemReqTag::A:
        f.a_packets.at(f.a_packet_idx).bytes = it->second.read_packet.bytes;
        ++f.a_packet_idx; break;
      case TmemReqTag::META:
        f.meta_packet.bytes = it->second.read_packet.bytes; break;
      case TmemReqTag::C:
        f.c_packets.at(f.c_packet_idx).bytes = it->second.read_packet.bytes;
        ++f.c_packet_idx; break;
      default: break;
      }
      completed_tmem_rsp_.erase(it);
    } else {
      // LMEM response handling (FETCH_B or FETCH_B_CHUNK).
      auto it = completed_lmem_rsp_.find(f.pending_req_id);
      if (it == completed_lmem_rsp_.end()) return;
      uint64_t addr = 0;
      if (f.state == FetchState::FETCH_B) {
        addr = f.b_lmem_base + f.b_packet_idx * kPacketBytes;
      } else {
        addr = f.b_chunk_base
             + uint64_t(f.b_chunk_idx) * f.b_chunk_bytes
             + uint64_t(f.b_packet_idx) * kPacketBytes;
      }
      core_->lmem_read(f.b_packets.at(f.b_packet_idx).data(), addr, kPacketBytes);
      ++f.b_packet_idx;
      completed_lmem_rsp_.erase(it);
    }
    f.pending_req_id = 0;
    f.pending_tag    = TmemReqTag::NONE;
  }

  switch (f.state) {
  case FetchState::FETCH_A: {
    if (read_one_a_packet(f)) {
      f.state = f.need_meta ? FetchState::FETCH_META
               : f.need_c   ? FetchState::FETCH_C
               :              FetchState::FETCH_B;
    }
    break;
  }
  case FetchState::FETCH_META: {
    if (read_meta_packet(f)) {
      f.state = f.need_c ? FetchState::FETCH_C : FetchState::FETCH_B;
    }
    break;
  }
  case FetchState::FETCH_C: {
    if (read_one_c_packet(f)) {
      f.state = FetchState::FETCH_B;
    }
    break;
  }
  case FetchState::FETCH_B: {
    if (read_one_b_packet(f)) {
      f.state = FetchState::CONVERT;
    }
    break;
  }
  case FetchState::FETCH_B_CHUNK: {
    if (read_one_b_chunk_packet(f)) {
      f.state = FetchState::CONVERT_B;
    }
    break;
  }
  case FetchState::CONVERT: {
    auto tile = build_converted_tile(f);
    if (tile) {
      DT(3, "OperandFetchStage: SOP tile wid=" << f.job.wid
         << " uuid=#" << f.job.uuid
         << " b_chunks=" << (int)f.total_b_chunks);
      Output.push(std::move(tile), config_.convert_latency);
      if (f.total_b_chunks > 1) {
        // More B chunks: start streaming subsequent chunks.
        f.b_chunk_idx = 1;
        f.b_packet_idx  = 0;
        f.b_packet_count = tmem_packet_count(f.job.fmt_b, false);
        f.b_packets.clear();
        f.b_packets.resize(f.b_packet_count);
        f.state = FetchState::FETCH_B_CHUNK;
      } else {
        f.reset();  // dense mode: single tile, done
      }
    }
    break;
  }
  case FetchState::CONVERT_B: {
    auto tile = build_b_only_tile(f);
    if (tile) {
      DT(3, "OperandFetchStage: B chunk " << (int)f.b_chunk_idx
         << "/" << (int)f.total_b_chunks << " wid=" << f.job.wid);
      // Push to Output; SimPort backpressure stalls if Stage3 Input is full.
      Output.push(std::move(tile), 1);
      ++f.b_chunk_idx;
      if (f.b_chunk_idx < f.total_b_chunks) {
        f.b_packet_idx  = 0;
        f.b_packet_count = tmem_packet_count(f.job.fmt_b, false);
        f.b_packets.clear();
        f.b_packets.resize(f.b_packet_count);
        f.state = FetchState::FETCH_B_CHUNK;
      } else {
        f.reset();  // all B chunks done
      }
    }
    break;
  }
  default: break;
  }
}

// ---------------------------------------------------------------------------
// Per-packet read helpers
// ---------------------------------------------------------------------------

bool OperandFetchStage::read_one_a_packet(PendingFetch& f) {
  if (f.a_packet_idx >= f.a_packet_count) return true;
  uint64_t req_id = issue_tmem_read(f.job.a_taddr, f.a_packet_idx);
  if (req_id == 0) return false;
  f.pending_req_id = req_id;
  f.pending_tag    = TmemReqTag::A;
  return false;
}

bool OperandFetchStage::read_meta_packet(PendingFetch& f) {
  uint32_t meta_pkt = (kSparseMetaOffset
                      + f.job.sparsity_meta_sel * kPacketBytes)
                      / kPacketBytes;
  uint64_t req_id = issue_tmem_read(f.job.a_taddr, meta_pkt);
  if (req_id == 0) return false;
  f.pending_req_id = req_id;
  f.pending_tag    = TmemReqTag::META;
  return false;
}

bool OperandFetchStage::read_one_c_packet(PendingFetch& f) {
  if (f.c_packet_idx >= f.c_packet_count) return true;
  uint64_t req_id = issue_tmem_read(f.job.d_taddr, f.c_packet_idx);
  if (req_id == 0) return false;
  f.pending_req_id = req_id;
  f.pending_tag    = TmemReqTag::C;
  return false;
}

bool OperandFetchStage::read_one_b_packet(PendingFetch& f) {
  if (f.b_packet_idx >= f.b_packet_count) return true;
  uint64_t addr = f.b_lmem_base + uint64_t(f.b_packet_idx) * kPacketBytes;
  uint64_t req_id = issue_lmem_read(addr);
  if (req_id == 0) return false;
  f.pending_req_id = req_id;
  return false;
}

bool OperandFetchStage::read_one_b_chunk_packet(PendingFetch& f) {
  if (f.b_packet_idx >= f.b_packet_count) return true;
  uint64_t addr = f.b_chunk_base
                + uint64_t(f.b_chunk_idx) * f.b_chunk_bytes
                + uint64_t(f.b_packet_idx) * kPacketBytes;
  uint64_t req_id = issue_lmem_read(addr);
  if (req_id == 0) return false;
  f.pending_req_id = req_id;
  return false;
}

// ---------------------------------------------------------------------------
// Precision conversion
// ---------------------------------------------------------------------------

std::shared_ptr<ConvertedTile> OperandFetchStage::build_converted_tile(
    PendingFetch& f) {
  auto tile = std::make_shared<ConvertedTile>();
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

  // Stream control: first (and possibly only) tile.
  tile->sop = 1;
  tile->eop = (f.total_b_chunks == 1) ? 1 : 0;
  tile->b_chunk_idx  = 0;
  tile->total_b_chunks = f.total_b_chunks;

  // A: 4 lines, each 8×8 → FP9.
  {
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

  // B: 4 lines → FP9.
  {
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

  // C: 4 subtiles → FP22.
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

  return tile;
}

std::shared_ptr<ConvertedTile> OperandFetchStage::build_b_only_tile(
    PendingFetch& f) {
  auto tile = std::make_shared<ConvertedTile>();

  tile->uuid = f.job.uuid;
  tile->wid  = f.job.wid;

  tile->fmt_a  = f.job.fmt_a;
  tile->fmt_b  = f.job.fmt_b;
  tile->fmt_c  = f.job.fmt_c;
  tile->fmt_d  = f.job.fmt_d;
  tile->shape_m = f.job.shape_m;
  tile->shape_n = f.job.shape_n;
  tile->sparsity_kind = f.job.sparsity_kind;
  tile->d_taddr = f.job.d_taddr;

  // Stream control (B-only, not SOP).
  tile->sop = 0;
  tile->eop = (f.b_chunk_idx + 1 == f.total_b_chunks) ? 1 : 0;
  tile->b_chunk_idx = f.b_chunk_idx;
  tile->total_b_chunks = f.total_b_chunks;

  // B: 4 lines → FP9.
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

  return tile;
}

}  // namespace vortex
