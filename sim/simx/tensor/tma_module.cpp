// TMA — full state machine with XFER_DATA per-packet loop.

#include "tma_module.h"
#include "core.h"
#include "debug.h"
#include "idescriptor.h"

namespace vortex {
namespace tensor {

Tma::Tma(const SimContext& ctx, const char* name, Core* core,
         const TmaConfig& config)
  : SimObject<Tma>(ctx, name)
  , Inputs(ISSUE_WIDTH, this)
  , Outputs(ISSUE_WIDTH, this)
  , CacheReq(this, 2)
  , CacheRsp(this, 2)
  , TmemReadReq(this, 2)
  , TmemReadRsp(this, 2)
  , TmemWriteReq(this, 2)
  , TmemWriteRsp(this, 2)
  , LmemDescReq(this, 2)
  , LmemDescRsp(this, 2)
  , AsyncCompletionOut(this, 2)
  , core_(core)
  , config_(config) {
  this->reset();
}

void Tma::reset() {
  active_.clear();
  cache_rsps_.clear(); lmem_rsps_.clear(); tmem_rsps_.clear();
  next_req_id_ = 1;
}

// ---------------------------------------------------------------------------
// tick
// ---------------------------------------------------------------------------

void Tma::tick() {
  // Drain responses into maps.
  if (!CacheRsp.empty())     { auto r = CacheRsp.front(); cache_rsps_[r.tag] = r; CacheRsp.pop(); }
  if (!LmemDescRsp.empty())  { auto r = LmemDescRsp.front(); lmem_rsps_[r.tag] = r; LmemDescRsp.pop(); }
  if (!TmemReadRsp.empty())  { auto r = TmemReadRsp.front(); tmem_rsps_[r.tag] = r; TmemReadRsp.pop(); }
  if (!TmemWriteRsp.empty()) { auto r = TmemWriteRsp.front(); tmem_rsps_[r.tag] = r; TmemWriteRsp.pop(); }

  // Accept new traces.
  decode_trace();

  // Advance active jobs.
  for (auto& job : active_) advance_job(job);

  // Remove completed.
  while (!active_.empty() && active_.front().stage == TmaStage::IDLE)
    active_.pop_front();
}

// ---------------------------------------------------------------------------
// Decode incoming traces
// ---------------------------------------------------------------------------

void Tma::decode_trace() {
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    if (Inputs[iw].empty() || active_.size() >= config_.max_inflight) continue;
    auto trace = Inputs[iw].front();
    auto* td = dynamic_cast<TmaTraceData*>(trace->data.get());
    if (td == nullptr) { Inputs[iw].pop(); Outputs[iw].push(trace, 3); continue; }
    Inputs[iw].pop();

    ActiveJob job;
    job.trace = trace; job.lane = iw;
    job.is_load         = (td->funct3 == 0b110);
    job.tensor_map_addr = td->tensor_map_addr;
    job.args_lmem_ptr   = static_cast<uint64_t>(td->args_lmem_ptr);
    job.complete_tx     = ((td->raw_funct7 >> 5) & 1) != 0;
    job.async_id        = next_req_id_++;
    job.taddr           = 0;  // filled from cpabulk_transfer_args
    job.stage           = TmaStage::READ_TRANSFER_ARGS;

    DT(3, "Tma: trace=#" << trace->uuid << (job.is_load ? " LD" : " ST")
       << " tmap=0x" << std::hex << job.tensor_map_addr
       << " args=0x" << job.args_lmem_ptr << std::dec);
    active_.push_back(std::move(job));
  }
}

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

void Tma::advance_job(ActiveJob& job) {
  // Check pending request — if any, try to consume response.
  if (job.pending_req_id != 0) {
    bool done = false;
    switch (job.pending_tag) {
    case ReqTag::DESC: {
      auto it = lmem_rsps_.find(job.pending_req_id);
      if (it != lmem_rsps_.end()) {
        // Functional read of cpabulk_transfer_args_t (32 B) from LMEM.
        core_->lmem_read(reinterpret_cast<uint8_t*>(&job.smem_addr),
                         job.args_lmem_ptr, 4);
        core_->lmem_read(reinterpret_cast<uint8_t*>(&job.mbar_addr),
                         job.args_lmem_ptr + 4, 4);
        core_->lmem_read(reinterpret_cast<uint8_t*>(job.coords),
                         job.args_lmem_ptr + 8, 20);
        lmem_rsps_.erase(it); done = true;
      }
    } break;
    case ReqTag::TMAP_LO: {
      auto it = cache_rsps_.find(job.pending_req_id);
      if (it != cache_rsps_.end()) {
        core_->dcache_read(job.tmap_buf, job.tensor_map_addr, 64);
        cache_rsps_.erase(it); done = true;
      }
    } break;
    case ReqTag::TMAP_HI: {
      auto it = cache_rsps_.find(job.pending_req_id);
      if (it != cache_rsps_.end()) {
        core_->dcache_read(job.tmap_buf + 64, job.tensor_map_addr + 64, 64);
        const auto* tmap = reinterpret_cast<const tensor_map_t*>(job.tmap_buf);
        job.global_addr = tmap->global_address;
        switch (tmap->element_type) {
        case 0:  job.elem_bytes = 2; break;  // CUDA_R_16F
        case 1:  job.elem_bytes = 4; break;  // CUDA_R_32F
        case 3:  case 6: job.elem_bytes = 1; break;  // CUDA_R_8I / CUDA_R_8U
        default: job.elem_bytes = 4; break;
        }
        uint64_t elems = 1;
        uint8_t rank = tmap->rank;
        if (rank == 0) rank = 1;
        for (int d = 0; d < rank; ++d)
          elems *= static_cast<uint32_t>(tmap->box_size[d]);
        job.tile_bytes = static_cast<uint32_t>(elems) * job.elem_bytes;
        if (job.tile_bytes == 0) job.tile_bytes = 64;
        job.total_pkts = (job.tile_bytes + 63) / 64;
        cache_rsps_.erase(it); done = true;
      }
    } break;
    case ReqTag::XFER_CACHE: {
      auto it = cache_rsps_.find(job.pending_req_id);
      if (it != cache_rsps_.end()) {
        if (job.is_load) {
          uint32_t epp = std::max<uint32_t>(1, 64 / job.elem_bytes);
          uint64_t dram_addr = job.compute_elem_addr(job.next_pkt * epp);
          core_->dcache_read(job.staging_pkt.bytes.data(), dram_addr, 64);
          // Issue TMEM write before consuming cache response — if port is full,
          // leave response in map and retry next tick.
          job.pending_tag = ReqTag::XFER_TMEM;
          if (!TmemWriteReq.full()) {
            TensorMemPortReq wr;
            wr.tag = job.pending_req_id;
            wr.access_type = TensorMemPortReq::AccessType::Write;
            wr.taddr = job.taddr;
            wr.packet_idx = job.next_pkt;
            wr.write_packet = job.staging_pkt;
            TmemWriteReq.push(wr, 0);
            cache_rsps_.erase(it); done = true;
          }
          // If TmemWriteReq full: cache response stays in map, retry next tick.
        } else {
          // ST: write staging data to DRAM via cache.
          uint32_t epp = std::max<uint32_t>(1, 64 / job.elem_bytes);
          uint64_t dram_addr = job.compute_elem_addr(job.next_pkt * epp);
          core_->dcache_write(job.staging_pkt.bytes.data(), dram_addr, 64);
          cache_rsps_.erase(it); done = true;
          ++job.next_pkt;
        }
      }
    } break;
    case ReqTag::XFER_TMEM: {
      auto it = tmem_rsps_.find(job.pending_req_id);
      if (it != tmem_rsps_.end()) {
        if (job.is_load) {
          // LD: TMEM write ack — advance to next packet.
          tmem_rsps_.erase(it); done = true;
          ++job.next_pkt;
        } else {
          // ST: TMEM read completed → staging_pkt has data. Issue cache write.
          job.staging_pkt = it->second.read_packet;
          tmem_rsps_.erase(it); done = true;
          job.pending_tag = ReqTag::XFER_CACHE;
          if (!CacheReq.full()) {
            MemReq wr(job.global_addr + job.next_pkt * 64, true, AddrType::Global,
                      job.pending_req_id);
            CacheReq.push(wr, 0);
            return;  // pending_req_id stays set; XFER_CACHE handles response
          }
          // CacheReq full: leave pending state, retry next tick.
          job.pending_req_id = 0; job.pending_tag = ReqTag::NONE;
        }
      }
    } break;
    default: break;
    }
    if (done) {
      job.pending_req_id = 0;
      job.pending_tag = ReqTag::NONE;
      // Response consumed — stop here. Next stage executes next tick.
      return;
    }
    // pending_req_id is set but response isn't ready yet — stall.
    if (job.pending_req_id != 0) return;
  }

  switch (job.stage) {
  case TmaStage::IDLE: break;

  case TmaStage::READ_TRANSFER_ARGS:
    if (LmemDescReq.full()) break;
    { LsuReq req(LSU_CHANNELS); req.tag = next_req_id_++; req.addrs.at(0) = job.args_lmem_ptr; req.mask.set(0);
      LmemDescReq.push(req, 0);
      job.pending_req_id = req.tag; job.pending_tag = ReqTag::DESC;
      DT(3, "Tma: DESC req_id=" << req.tag); }
    job.stage = TmaStage::READ_TENSOR_MAP_LO;
    break;

  case TmaStage::READ_TENSOR_MAP_LO:
    if (CacheReq.full()) break;
    { MemReq mr(job.tensor_map_addr, false, AddrType::Global, next_req_id_);
      CacheReq.push(mr, 0);
      job.pending_req_id = next_req_id_++; job.pending_tag = ReqTag::TMAP_LO;
      DT(3, "Tma: TMAP_LO tag=" << mr.tag);
      job.stage = TmaStage::READ_TENSOR_MAP_HI; }
    break;

  case TmaStage::READ_TENSOR_MAP_HI:
    if (CacheReq.full()) break;
    { MemReq mr(job.tensor_map_addr + 64, false, AddrType::Global, next_req_id_);
      CacheReq.push(mr, 0);
      job.pending_req_id = next_req_id_++; job.pending_tag = ReqTag::TMAP_HI;
      DT(3, "Tma: TMAP_HI tag=" << mr.tag);
      job.stage = TmaStage::XFER_DATA; }
    break;

  case TmaStage::XFER_DATA: {
    if (job.next_pkt >= job.total_pkts) { job.stage = TmaStage::COMPLETE; break; }

    uint32_t elems_per_pkt = std::max<uint32_t>(1, 64 / job.elem_bytes);
    uint32_t start_elem = job.next_pkt * elems_per_pkt;
    uint64_t dram_addr = job.compute_elem_addr(start_elem);

    if (job.is_load) {
      if (CacheReq.full()) break;
      MemReq mr(dram_addr, false, AddrType::Global, next_req_id_);
      CacheReq.push(mr, 0);
      job.pending_req_id = next_req_id_++; job.pending_tag = ReqTag::XFER_CACHE;
      DT(4, "Tma: LD pkt=" << job.next_pkt << "/" << job.total_pkts
         << " dram=0x" << std::hex << dram_addr << std::dec);
    } else {
      // ST: issue TMEM read for current packet.
      if (TmemReadReq.full()) break;
      TensorMemPortReq rr;
      rr.tag = next_req_id_++;
      rr.access_type = TensorMemPortReq::AccessType::Read;
      rr.taddr = job.taddr;
      rr.packet_idx = job.next_pkt;
      TmemReadReq.push(rr, 0);
      job.pending_req_id = rr.tag; job.pending_tag = ReqTag::XFER_TMEM;
      DT(4, "Tma: ST pkt=" << job.next_pkt << "/" << job.total_pkts);
    }
    break;
  }

  case TmaStage::COMPLETE:
    complete_job(job);
    break;
  }
}

void Tma::complete_job(ActiveJob& job) {
  TensorAsyncOpCompletion done;
  done.async_id = job.async_id;
  if (!AsyncCompletionOut.full()) AsyncCompletionOut.push(done, 1);
  Outputs[job.lane].push(job.trace, 0);
  job.stage = TmaStage::IDLE;
  DT(3, "Tma: complete trace=#" << job.trace->uuid);
}

}  // namespace tensor
}  // namespace vortex
