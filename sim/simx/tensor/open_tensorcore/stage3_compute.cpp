// Stage3 ComputePipeline implementation.
//
// Drives the TensorCoreTop 8×8 compute array with primitives from internal
// operand SRAMs (AMem/BMem/CMem), accumulates results in DMem, and writes the
// completed D tile back to TMEM.

#include "stage3_compute.h"

#include <cstring>

#include "core.h"
#include "debug.h"
#include "open_tensorcore/tensor_compute/fp_types.h"
#include "open_tensorcore/tensor_compute/fp22_to_fp16.h"
#include "open_tensorcore/tensor_compute/sparse_select.h"
#include "tensor_cfg.h"

namespace vortex {

namespace vt = vortex::tensor;

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

uint32_t ComputePipeline::accum_phase_count(const uint32_t sparsity_kind) {
  if (sparsity_kind == vt::sparse_none) {
    return kDenseAccumPhases;
  }
  return kSparseAccumPhases;
}

uint32_t ComputePipeline::mem_k_phase(const uint32_t sparsity_kind,
                                      const uint32_t accum_phase) {
  return (sparsity_kind == vt::sparse_none) ? (accum_phase / 2)
                                            : accum_phase;
}

uint32_t ComputePipeline::fmt_element_bytes(uint32_t fmt) {
  switch (fmt) {
  case vt::fp8::id:  return 1;
  case vt::fp16::id: return 2;
  case vt::fp32::id: return 4;
  default:            return 0;
  }
}

// ---------------------------------------------------------------------------
// Constructor / reset / tick
// ---------------------------------------------------------------------------

ComputePipeline::ComputePipeline(const SimContext& ctx, const char* name,
                                 Core* core, DMem* dmem, const ComputeConfig& config)
  : SimObject<ComputePipeline>(ctx, name)
  , Input(this, config.input_depth)
  , Output(this, config.output_depth)   // sink endpoint: OTC reads via front()/pop()
  , TmemWriteReq(this, 0)               // source/relay port (→ TMEM write arbiter)
  , TmemWriteRsp(this, config.tmem_wrsp_depth)
  , core_(core)
  , config_(config)
  , dmem_(dmem) {
  this->reset();
}

void ComputePipeline::reset() {
  amem_.clear();
  bmem_[0].clear();
  bmem_[1].clear();
  cmem_.clear();
  dmem_->clear();
  tensorcore_.reset();
  pending_.clear();
  completed_tmem_wrsp_.clear();
  next_request_id_ = 1;
  staged_sparse_meta_.fill(0);
  staged_has_sparse_meta_ = false;

  // Clear and reindex collector buffers.
  abuf_.clear();
  mbuf_.clear();
  bbuf_[0] = BBuf(0); bbuf_[0].clear();
  bbuf_[1] = BBuf(1); bbuf_[1].clear();
  bbuf_[2] = BBuf(2); bbuf_[2].clear();
  bbuf_[3] = BBuf(3); bbuf_[3].clear();
}

void ComputePipeline::tick() {
  drain_tmem_responses();

  // Accept new job if idle.
  if (!Input.empty() && pending_.empty()) {
    auto tile = Input.front();
    Input.pop();
    ActiveJob job;
    job.tile  = std::move(tile);
    job.state = ComputeState::FILL;
    DT(3, "ComputePipeline: start wid=" << job.tile->wid
       << " uuid=#" << job.tile->uuid
       << " shape=" << job.tile->shape_m << "x" << job.tile->shape_n
       << " sp=" << static_cast<uint32_t>(job.tile->sparsity_kind));
    pending_.push_back(std::move(job));
  }

  if (pending_.empty()) return;

  auto& job = pending_.front();
  switch (job.state) {
  case ComputeState::FILL:
    fill_abc(job);
    job.active_bmem = 0;
    job.b_chunk_recv = 1;   // B0 from SOP tile
    job.total_b_chunks = job.tile->total_b_chunks;
    job.round_issued_subtiles = 0;
    if (job.tile->sparsity_kind == vt::sparse_none) {
      job.phases_per_round = kDenseAccumPhases;  // 4 (2 K-phases x 2 bubble)
    } else {
      job.phases_per_round = 2;  // sparse: 2 K-phases, no bubble
    }
    job.state = ComputeState::COMPUTE;
    break;

  case ComputeState::COMPUTE: {
    // Step order: compute first, so swap can catch round_issued_subtiles→4
    // in the same tick, and receive can immediately fill the newly-idle bank.

    // C: Drive TensorCoreTop compute (may set round_issued_subtiles → kSubtiles).
    advance_compute(job);

    // B: Try to swap to next K round (sees round_issued_subtiles just set).
    try_swap_to_next_round(job);

    // A: Try to receive next B chunk from Stage2a (idle bank freed by swap).
    try_receive_next_chunk(job);

    // D: Check all done.
    bool all_chunks_done = (job.b_chunk_recv == job.total_b_chunks);
    bool all_issued_done = (job.round_issued_subtiles == kSubtiles);
    if (all_chunks_done && all_issued_done
        && job.final_retired_subtiles == kSubtiles) {
      if (build_store_packets(job)) {
        job.state = ComputeState::STORE;
      }
    }
    break;
  }

  case ComputeState::STORE:
    // Check pending write response.
    if (job.pending_wrsp_id != 0) {
      auto it = completed_tmem_wrsp_.find(job.pending_wrsp_id);
      if (it == completed_tmem_wrsp_.end()) break;  // stall
      completed_tmem_wrsp_.erase(it);
      job.pending_wrsp_id = 0;
      ++job.d_packet_idx;
    }
    if (advance_store(job)) {
      job.state = ComputeState::DONE;
    }
    break;

  case ComputeState::DONE:
    finish_job(job);
    pending_.pop_front();
    break;

  default:
    break;
  }
}

// ---------------------------------------------------------------------------
// FILL — copy ConvertedTile data into operand SRAMs
// ---------------------------------------------------------------------------

void ComputePipeline::fill_abc(ActiveJob& job) {
  const auto& tile = *job.tile;

  uint8_t  cb  = tile.collector_buffer;
  int8_t   bbi = tile.bbuf_idx;
  bool use_abuf = (bbi < 0) && (cb != 0x3);   // ws=0, not DISCARD
  bool use_bbuf = (bbi >= 0) && (cb != 0x3);   // ws=1, not DISCARD

  // Clear shared resources.
  cmem_.clear();
  dmem_->clear();
  tensorcore_.reset();

  // A operand section.
  if (use_abuf && cb == 0x0) {
    // FILL abuf: write A data to abuf, mark fill.
    abuf_.clear();
    for (uint32_t line = 0; line < AMem::kDepth; ++line) {
      uint32_t k_phase = line / 2, m_block = line % 2;
      uint16_t data[8][8];
      for (uint32_t i = 0; i < 8; ++i)
        for (uint32_t k = 0; k < 8; ++k)
          data[i][k] = tile.a_fp9[m_block * 8 + i][k_phase * 8 + k];
      abuf_.write_line(line, data);
    }
    abuf_.fill(job.tile->wid);
    // Also fill mbuf if sparse.
    if (tile.has_sparse_meta) {
      mbuf_.clear();
      mbuf_.write(tile.sparse_meta);
      mbuf_.fill(job.tile->wid);
    }
  } else if (use_abuf && (cb == 0x1 || cb == 0x2)) {
    // USE/LASTUSE: A already in abuf, mark compute started.
    abuf_.mark_compute_started();
    if (mbuf_.valid()) mbuf_.mark_compute_started();
  } else {
    // DISCARD or bbuf mode: normal AMem fill.
    amem_.clear();
    for (uint32_t line = 0; line < AMem::kDepth; ++line) {
      uint32_t k_phase = line / 2, m_block = line % 2;
      uint16_t data[8][8];
      for (uint32_t i = 0; i < 8; ++i)
        for (uint32_t k = 0; k < 8; ++k)
          data[i][k] = tile.a_fp9[m_block * 8 + i][k_phase * 8 + k];
      amem_.write_converted_line(line, data);
    }
  }

  // B operand section.
  if (use_bbuf && cb == 0x0) {
    // FILL bbuf: write B data to bbuf[bbi], mark fill.
    bbuf_[bbi].clear();
    for (uint32_t line = 0; line < BMem::kDepth; ++line) {
      uint32_t k_phase = line / 2, n_block = line % 2;
      uint16_t data[8][8];
      for (uint32_t k = 0; k < 8; ++k)
        for (uint32_t j = 0; j < 8; ++j)
          data[k][j] = tile.b_fp9[k_phase * 8 + k][n_block * 8 + j];
      bbuf_[bbi].write_line(line, data);
    }
    bbuf_[bbi].fill(job.tile->wid);
  } else if (use_bbuf && (cb == 0x1 || cb == 0x2)) {
    // USE/LASTUSE: B already in bbuf, mark compute started.
    bbuf_[bbi].mark_compute_started();
  } else {
    // DISCARD or abuf mode: normal BMem fill.
    bmem_[0].clear();
    for (uint32_t line = 0; line < BMem::kDepth; ++line) {
      uint32_t k_phase = line / 2, n_block = line % 2;
      uint16_t data[8][8];
      for (uint32_t k = 0; k < 8; ++k)
        for (uint32_t j = 0; j < 8; ++j)
          data[k][j] = tile.b_fp9[k_phase * 8 + k][n_block * 8 + j];
      bmem_[0].write_converted_line(line, data);
    }
  }

  // Fill CMem: 4 subtiles, each 8×8 FP22 block.
  if (tile.has_c) {
    for (uint32_t subtile = 0; subtile < CMem::kDepth; ++subtile) {
      uint32_t m_block = subtile / 2;
      uint32_t n_block = subtile % 2;
      uint32_t data[8][8];
      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
          data[i][j] = tile.c_fp22[m_block * 8 + i][n_block * 8 + j];
        }
      }
      cmem_.write_converted_subtile(subtile, data);
    }
  }

  // Always stage sparse metadata for the compute path.
  staged_has_sparse_meta_ = tile.has_sparse_meta;
  if (tile.has_sparse_meta) {
    staged_sparse_meta_ = tile.sparse_meta;
  }
}

// ---------------------------------------------------------------------------
// COMPUTE — issue primitives, drive TensorCoreTop, accumulate into DMem
// ---------------------------------------------------------------------------

void ComputePipeline::advance_compute(ActiveJob& job) {
  const auto& tile = *job.tile;
  const uint32_t sp_kind = tile.sparsity_kind;
  const uint32_t phases  = accum_phase_count(sp_kind);

  // Try to issue one primitive if there are more to issue and tensorcore is ready.
  //
  // 16×16 MMA → 4 output subtiles (sm=0/1, sn=0/1). Each subtile needs
  // 'phases' accum passes through TensorCore for K=16 accumulation:
  //   dense: 4 phases (K=16 split into 2 K phases × 2 pipeline bubbles)
  //   sparse: 2 phases (K=16 split into 2 K phases × 1, no bubble)
  if (job.issue_subtile < kSubtiles && tensorcore_.ready(true)) {
    const uint32_t accum = job.issue_accum_phase;
    // mem_k_phase: same K-phase data reused for consecutive phases (dense bubble).
    const uint32_t mk = mem_k_phase(sp_kind, accum);
    const uint32_t sm = job.issue_subtile / 2;  // M block index (0=top, 1=bottom)
    const uint32_t sn = job.issue_subtile % 2;  // N block index (0=left, 1=right)

    uint16_t a[kPrimitiveDim][kPrimitiveDim] = {};
    uint16_t b[kPrimitiveDim][kPrimitiveDim] = {};
    uint32_t c[kPrimitiveDim][kPrimitiveDim] = {};

    // Read A: from abuf if valid and in abuf mode, else AMem.
    uint8_t  cb  = tile.collector_buffer;
    int8_t   bbi = tile.bbuf_idx;
    if (abuf_.valid() && bbi < 0 && cb != 0x3) {
      abuf_.read_primitive(mk * 2 + sm, a);
    } else {
      amem_.read_primitive(mk * 2 + sm, a, false);
    }

    // Read B: from bbuf if valid and in bbuf mode, else BMem.
    if (bbi >= 0 && cb != 0x3 && bbuf_[bbi].valid()) {
      bbuf_[bbi].read_primitive(mk * 2 + sn, b);
    } else {
      bmem_[job.active_bmem].read_primitive(mk * 2 + sn, b, false);
    }

    // C bypass only in the first accum phase (non-resident accumulator mode).
    if (accum == 0 && tile.has_c) {
      cmem_.read_subtile_fp22(job.issue_subtile, c);
    }

    // Build sparse row metadata for the select pipe.
    uint16_t sparse_row_meta[kPrimitiveDim] = {};
    if (sp_kind != vt::sparse_none && staged_has_sparse_meta_) {
      uint32_t meta_line = sm * 2 + mk;  // 4 meta rows, same indexing as AMem lines
      const uint8_t* meta_base = staged_sparse_meta_.data() + meta_line * 16;
      for (uint32_t row = 0; row < kPrimitiveDim; ++row) {
        sparse_row_meta[row] = vortex::sparse::row_meta_bits(meta_base, row);
      }
    }

    TensorCoreMeta meta;
    meta.wid            = tile.wid;
    meta.async_id       = 0;
    meta.c_subtile_id   = job.issue_subtile;
    meta.accum_phase_id = accum;
    meta.valid          = true;

    tensorcore_.push_uop(a, b, c, meta, sp_kind, sparse_row_meta);

    // Advance: cycle through accum phases, then advance subtile.
    if (accum + 1 < phases) {
      ++job.issue_accum_phase;
    } else {
      job.issue_accum_phase = 0;
      ++job.issue_subtile;
    }

    // Track per-round issue progress for ping-pong swap.
    if (accum + 1 == phases) {
      ++job.round_issued_subtiles;
    }
  }

  // Tick the compute array every cycle (backpressure handled via ready()).
  tensorcore_.tick(true);

  // Collect retired primitives (at most one per tick).
  TensorCoreRetire retired;
  if (tensorcore_.pop_retired(&retired)) {
    uint32_t subtile = retired.meta.c_subtile_id;
    if (retired.meta.accum_phase_id == 0) {
      dmem_->write_subtile_fp22(subtile, retired.fp22_out);
    } else {
      dmem_->accumulate_subtile_fp22(subtile, retired.fp22_out);
    }
    if (retired.meta.accum_phase_id == (phases - 1)) {
      ++job.final_retired_subtiles;
    }
  }
}

// ---------------------------------------------------------------------------
// Ping-pong helpers
// ---------------------------------------------------------------------------

void ComputePipeline::fill_b_bank(uint32_t bank_idx,
                                  const std::shared_ptr<ConvertedTile>& tile) {
  bmem_[bank_idx].clear();
  for (uint32_t line = 0; line < BMem::kDepth; ++line) {
    uint32_t k_phase = line / 2;
    uint32_t n_block = line % 2;
    uint16_t data[8][8];
    for (uint32_t k = 0; k < 8; ++k) {
      for (uint32_t j = 0; j < 8; ++j) {
        data[k][j] = tile->b_fp9[k_phase * 8 + k][n_block * 8 + j];
      }
    }
    bmem_[bank_idx].write_converted_line(line, data);
  }
}

bool ComputePipeline::try_receive_next_chunk(ActiveJob& job) {
  if (job.b_chunk_recv >= job.total_b_chunks) return false;
  if (bmem_[(job.active_bmem ^ 1)].valid()) return false;  // idle bank has old data
  if (Input.empty()) return false;

  auto tile = Input.front();
  if (tile->b_chunk_idx != job.b_chunk_recv) return false;  // validate sequence
  Input.pop();
  fill_b_bank((job.active_bmem ^ 1), tile);  // write to idle bank

  DT(4, "ComputePipeline: recv B chunk " << (int)tile->b_chunk_idx
     << "/" << (int)job.total_b_chunks << " wid=" << job.tile->wid);
  return true;
}

bool ComputePipeline::try_swap_to_next_round(ActiveJob& job) {
  if (job.round_issued_subtiles < kSubtiles) return false;
  if (job.b_chunk_recv >= job.total_b_chunks) return false;
  if (!bmem_[(job.active_bmem ^ 1)].valid()) return false;

  // Invalidate the bank we are vacating so it can be reused.
  bmem_[job.active_bmem].clear();
  job.active_bmem ^= 1;
  ++job.b_chunk_recv;
  job.round_issued_subtiles = 0;

  DT(4, "ComputePipeline: swap to BMem[" << job.active_bmem
     << "] round " << job.b_chunk_recv << "/" << job.total_b_chunks);
  return true;
}

// ---------------------------------------------------------------------------
// STORE — build TMEM-format packets and write D back
// ---------------------------------------------------------------------------

bool ComputePipeline::build_store_packets(ActiveJob& job) {
  const auto& tile = *job.tile;
  uint32_t elem_bytes = fmt_element_bytes(tile.fmt_d);
  if (elem_bytes == 0) return false;

  uint32_t M = tile.shape_m;
  uint32_t N = tile.shape_n;

  // D TMEM layout — must match the scalar tcgen05.ld readback the kernel uses:
  //   out_index[r*W + lane] = tcu_ld(d_taddr | (r*4)<<16)  for r in [0..cols),
  //                                                            lane in [0..W)
  // i.e. the warp (W lanes) reads the D element with linear index
  //   idx = r*W + lane  (row-major over M×N, idx = m*N + n)
  // from TMEM cell (row = lane = idx % W, bank = r = idx / W). Each cell is one
  // 4-byte bank slot at byte offset (idx%W)*kTmemRowBytes + (idx/W)*4.
  constexpr uint32_t kWarpLanes = 32;     // readback lanes (= NUM_THREADS)
  constexpr uint32_t kTmemRowBytes = 128; // 32 banks × 4 B per TMEM row
  uint32_t max_boff = 0;
  for (uint32_t idx = 0; idx < M * N; ++idx) {
    uint32_t boff = (idx % kWarpLanes) * kTmemRowBytes + (idx / kWarpLanes) * 4;
    if (boff + 4 > max_boff) max_boff = boff + 4;
  }
  uint32_t total_pkts = (max_boff + 64 - 1) / 64;
  job.d_store_packets.assign(total_pkts, TmemPacket{});

  for (uint32_t subtile = 0; subtile < kSubtiles; ++subtile) {
    uint32_t fp22[kPrimitiveDim][kPrimitiveDim];
    dmem_->read_subtile_fp22(subtile, fp22);

    uint32_t sm = subtile / 2;
    uint32_t sn = subtile % 2;

    for (uint32_t i = 0; i < kPrimitiveDim; ++i) {
      uint32_t gr = sm * kPrimitiveDim + i;   // M direction (row)
      for (uint32_t j = 0; j < kPrimitiveDim; ++j) {
        uint32_t gc = sn * kPrimitiveDim + j;  // N direction (col)
        if (gr >= M || gc >= N) continue;

        // FP22 → output format.
        uint32_t bits = 0;
        switch (tile.fmt_d) {
        case vt::fp32::id: bits = fp22_to_fp32(fp22[i][j]); break;
        case vt::fp16::id: bits = fp22_to_fp16(fp22[i][j]); break;
        case vt::fp8::id:  bits = fp22_to_fp8_e4m3(fp22[i][j]); break;
        default: return false;
        }

        uint32_t idx = gr * N + gc;           // row-major D linear index
        uint32_t tile_boff = (idx % kWarpLanes) * kTmemRowBytes
                           + (idx / kWarpLanes) * 4;
        uint32_t dst_pkt = tile_boff / 64;
        uint32_t dst_off = tile_boff % 64;
        std::memcpy(job.d_store_packets[dst_pkt].bytes.data() + dst_off,
                    &bits, elem_bytes);
      }
    }
  }

  return true;
}

// ---------------------------------------------------------------------------
// TMEM write helpers
// ---------------------------------------------------------------------------

void ComputePipeline::drain_tmem_responses() {
  // One pop per cycle (SimPort allows a single dequeue per port per tick).
  if (!TmemWriteRsp.empty()) {
    auto rsp = TmemWriteRsp.front();
    TmemWriteRsp.pop();
    completed_tmem_wrsp_[rsp.request_id] = rsp;
  }
}

uint64_t ComputePipeline::issue_tmem_write(uint32_t taddr, uint32_t pkt_idx,
                                           const TmemPacket& data) {
  if (TmemWriteReq.full()) return 0;

  TensorMemPortReq req;
  req.request_id            = next_request_id_++;
  req.access_type           = TensorMemPortReq::AccessType::Write;
  req.taddr    = taddr;
  req.packet_idx = pkt_idx;
  req.write_packet          = data;

  DT(4, "ComputePipeline: tmem write req_id=" << req.request_id
     << " taddr=0x" << std::hex << taddr << std::dec
     << " pkt=" << pkt_idx);

  TmemWriteReq.push(req, 0);
  return req.request_id;
}

bool ComputePipeline::advance_store(ActiveJob& job) {
  if (job.d_packet_idx >= job.d_store_packets.size()) return true;

  uint64_t req_id = issue_tmem_write(job.tile->d_taddr,
                                     job.d_packet_idx,
                                     job.d_store_packets[job.d_packet_idx]);
  if (req_id == 0) return false;  // port full, retry next tick
  job.pending_wrsp_id = req_id;
  return false;
}

// ---------------------------------------------------------------------------
// DONE — emit completion signal
// ---------------------------------------------------------------------------

void ComputePipeline::finish_job(ActiveJob& job) {
  // Collector buffer cleanup.
  const auto& tile = *job.tile;
  if (tile.collector_buffer == 0x2 || tile.collector_buffer == 0x3) {
    // LASTUSE or DISCARD: invalidate after compute completes.
    if (tile.bbuf_idx < 0) {
      abuf_.invalidate();
      mbuf_.invalidate();
    } else {
      bbuf_[tile.bbuf_idx].invalidate();
    }
  } else {
    // FILL or USE: mark compute done (no longer inflight).
    if (tile.bbuf_idx < 0) {
      abuf_.mark_compute_done();
      mbuf_.mark_compute_done();
    } else {
      bbuf_[tile.bbuf_idx].mark_compute_done();
    }
  }

  CompletedMmaJob done;
  done.uuid    = job.tile->uuid;
  done.success = true;

  DT(3, "ComputePipeline: done wid=" << job.tile->wid
     << " uuid=#" << done.uuid);

  Output.push(done, 1);
  job.reset();
}

}  // namespace vortex
