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

#pragma once

#include <array>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <simobject.h>
#include <tensor_cfg.h>
#include "types.h"
#include "emulator.h"
#include "pipeline.h"
#include "cache_sim.h"
#include "local_mem.h"
#include "ibuffer.h"
#include "scoreboard.h"
#include "tensor_mem_port_types.h"
#include "tma.h"
#include "tmem.h"
#include "tmem_system.h"

#ifdef EXT_V_ENABLE
#include "voperands.h"
#include "vec_unit.h"
#else
#include "operands.h"
#endif

#include "dispatcher.h"
#include "func_unit.h"
#include "mem_coalescer.h"
#include "VX_config.h"

namespace vortex {

struct TmemWindowPlan;

class Socket;
class Arch;
class DCRS;

class Core : public SimObject<Core> {
public:
  struct PerfStats {
    uint64_t cycles;
    uint64_t instrs;
    uint64_t sched_idle;
    uint64_t sched_stalls;
    uint64_t ibuf_stalls;
    uint64_t scrb_stalls;
    uint64_t opds_stalls;
    uint64_t scrb_alu;
    uint64_t scrb_fpu;
    uint64_t scrb_lsu;
    uint64_t scrb_sfu;
    uint64_t scrb_csrs;
    uint64_t scrb_wctl;
  #ifdef EXT_V_ENABLE
    uint64_t vinstrs;
    uint64_t scrb_vpu;
  #endif
  #ifdef EXT_TCU_ENABLE
    uint64_t scrb_tcu;
  #endif
    uint64_t ifetches;
    uint64_t loads;
    uint64_t stores;
    uint64_t ifetch_latency;
    uint64_t load_latency;
    uint64_t stall_wait_barrier;
    uint64_t stall_tmem_read_port_busy;
    uint64_t stall_tmem_write_port_busy;
    uint64_t tma_load_count;
    uint64_t tma_load_latency_sum;
    uint64_t tma_store_count;
    uint64_t tma_store_latency_sum;
    uint64_t tmem_read_packets;
    uint64_t tmem_write_packets;

    PerfStats()
      : cycles(0)
      , instrs(0)
      , sched_idle(0)
      , sched_stalls(0)
      , ibuf_stalls(0)
      , scrb_stalls(0)
      , opds_stalls(0)
      , scrb_alu(0)
      , scrb_fpu(0)
      , scrb_lsu(0)
      , scrb_sfu(0)
      , scrb_csrs(0)
      , scrb_wctl(0)
    #ifdef EXT_V_ENABLE
      , vinstrs(0)
      , scrb_vpu(0)
    #endif
    #ifdef EXT_TCU_ENABLE
      , scrb_tcu(0)
    #endif
      , ifetches(0)
      , loads(0)
      , stores(0)
      , ifetch_latency(0)
      , load_latency(0)
      , stall_wait_barrier(0)
      , stall_tmem_read_port_busy(0)
      , stall_tmem_write_port_busy(0)
      , tma_load_count(0)
      , tma_load_latency_sum(0)
      , tma_store_count(0)
      , tma_store_latency_sum(0)
      , tmem_read_packets(0)
      , tmem_write_packets(0)
    {}
  };

  std::vector<SimPort<MemReq>> icache_req_ports;
  std::vector<SimPort<MemRsp>> icache_rsp_ports;

  std::vector<SimPort<MemReq>> dcache_req_ports;
  std::vector<SimPort<MemRsp>> dcache_rsp_ports;

  Core(const SimContext& ctx,
       uint32_t core_id,
       Socket* socket,
       const Arch &arch,
       const DCRS &dcrs
  );

  ~Core();

  void reset();

  void tick();

  void attach_ram(RAM* ram);
#ifdef VM_ENABLE
  void set_satp(uint64_t satp);
#endif

  bool running() const;

  void suspend(uint32_t wid, WarpStallReason reason);

  void resume(uint32_t wid);

  void set_stall_reason(uint32_t wid, WarpStallReason reason);

  bool barrier(uint32_t bar_id, uint32_t count, uint32_t wid);

  bool wspawn(uint32_t num_warps, Word nextPC);

  uint32_t id() const {
    return core_id_;
  }

  const Arch& arch() const {
    return arch_;
  }

  Socket* socket() const {
    return socket_;
  }

  const LocalMem::Ptr& local_mem() const {
    return local_mem_;
  }

  const MemCoalescer::Ptr& mem_coalescer(uint32_t idx) const {
    return mem_coalescers_.at(idx);
  }

  void dcache_read(void* data, uint64_t addr, uint32_t size) {
    return emulator_.dcache_read(data, addr, size);
  }

  void dcache_write(const void* data, uint64_t addr, uint32_t size) {
    return emulator_.dcache_write(data, addr, size);
  }

  // ==========================================================================
  // LMEM byte-addressable read/write helpers.
  //
  // `addr` is the absolute LMEM address (i.e., the value the kernel would put
  // into a pointer in the LMEM_BASE_ADDR..LMEM_BASE_ADDR+size range). Internally
  // we offset by LMEM_BASE_ADDR and call LocalMem::read/write directly.
  //
  // Used by tcgen05 control-plane paths that need to read PTX descriptors
  // out of shared memory: TC_decode reads operand_block_t and b_sdesc 64-bit;
  // mbarrier paths read/write 8 B mbarrier_state_t at a kernel-specified addr;
  // CPABULK_TENSOR_LD/ST reads the 32 B cpabulk_transfer_args_t.
  //
  // These are functional (byte-accurate) reads; timing of LMEM bank conflicts
  // is still produced by the SimPort traffic in the standard read/write
  // dispatch path used by load/store instructions.
  // ==========================================================================
  void lmem_read(void* data, uint64_t addr, uint32_t size) {
    auto offset = addr - LMEM_BASE_ADDR;
    local_mem_->read(data, offset, size);
  }

  void lmem_write(const void* data, uint64_t addr, uint32_t size) {
    auto offset = addr - LMEM_BASE_ADDR;
    local_mem_->write(data, offset, size);
  }

#ifdef EXT_TCU_ENABLE
  TensorUnit::Ptr& tensor_unit() {
    return tensor_unit_;
  }

  using TmemPacket = vortex::TmemPacket;
  using TmemRequestDesc = vortex::Tmem::PortRequestDesc;
  using TmemRequestKind = vortex::Tmem::PortRequestKind;
  using TmemTaddrBlockReason = vortex::TmemTaddrBlockReason;

  static constexpr uint32_t kTmemPayloadCols = Tmem::kPayloadCols;
  static constexpr uint32_t kTmemMetaCols = Tmem::kMetaCols;
  static constexpr uint32_t kTmemMetaColBase = Tmem::kMetaColBase;
  static constexpr uint32_t kTmemPacketBytes = Tmem::kPacketBytes;
  static constexpr uint32_t kTmemReadPacketsPerCycle = Tmem::kReadPacketsPerCycle;
  static constexpr uint32_t kTmemWritePacketsPerCycle = Tmem::kWritePacketsPerCycle;
  uint32_t tmem_alloc(uint32_t col_span, uint32_t reserved_operand = kTmemAllocReservedOperand);
  // PTX tcgen05.dealloc.
  bool tmem_dealloc(uint32_t taddr);
  void tmem_rel_permit();
  uint32_t tmem_shift(uint32_t wid, uint32_t taddr, uint32_t control_word = kTcuTmemOpControlPtxOnly);

  // ===== tcgen05 / cp.async.bulk.tensor data-plane Core API (Phase-2) =====
  //
  // cp.async.bulk.tensor (DRAM -> shared)
  //   tensor_map_addr : DRAM address of 128 B tensor_map_t (PTX §6.4.10.4)
  //   args_lmem_ptr   : LMEM address of 32 B cpabulk_transfer_args_t holding
  //                     {smem_addr, mbar_addr, coords[5], reserved}
  //   complete_tx     : when 1, on completion calls mbarrier_complete_tx
  //                     (args.mbar_addr, bytes_transferred)
  //   Returns the async_id used to track this transfer.
  uint32_t cpabulk_tensor_load(uint32_t wid,
                               uint64_t tensor_map_addr,
                               uint64_t args_lmem_ptr,
                               bool complete_tx);
  uint32_t cpabulk_tensor_store(uint32_t wid,
                                uint64_t tensor_map_addr,
                                uint64_t args_lmem_ptr);

  // tcgen05.cp (shared -> TMEM)
  //   taddr   : TMEM destination address
  //   s_desc_lmem_ptr : LMEM address holding 64-bit s_desc value
  //   shape   : tcgen05.cp .shape modifier (PTX §9.7.16.5.3)
  //   decompress : decompress modifier (none/b6x16_p32/b4x16_p64)
  bool tmem_cp(uint32_t wid,
               uint32_t taddr,
               uint64_t s_desc_lmem_ptr,
               uint32_t shape,
               uint32_t decompress);

  // tcgen05.ld / tcgen05.st (Phase-3.3.1 GAP-4): per-thread byte-range TMEM R/W.
  // Phase-3.4 Stage 0: PTX TADDR `[15:0]=lane, [31:16]=col_byte`. The
  // 'taddr' arg here is the col_base (= TmemAllocation key); byte_offset is
  // computed from PTX taddr by callers as
  //   byte_offset = (actual_lane - col_base) * kColBytes + col_byte
  // where actual_lane = taddr.lane + thread_id (warp-collective).
  bool tmem_taddr_read_bytes(uint32_t taddr, uint32_t byte_offset, uint8_t* dst, uint32_t bytes);
  bool tmem_taddr_write_bytes(uint32_t taddr, uint32_t byte_offset, const uint8_t* src, uint32_t bytes);
  bool tmem_find_allocation_by_lane(uint32_t lane, uint32_t* col_base) const;

  // tcgen05.fence::{before,after}_thread_sync
  bool mbar_fence(uint32_t wid, TcuFenceMode mode);

  // tcgen05.commit (PTX §9.7.16.5.7) -- registers inflight tcgen05.async ops
  // to mbar's expected_arrival_count; each completion triggers mbarrier_arrive.
  uint32_t mbar_commit(uint32_t wid,
                       uint64_t mbar_addr,
                       uint32_t cta_mask = 0);

  // tcgen05.wait::ld / tcgen05.wait::st -- block warp until prior tcgen05
  // load / store ops issued from this warp are complete.
  void issue_tcgen05_ld_async(uint32_t wid,
                              uint64_t trace_id,
                              const RegOpd& dst,
                              const ThreadMask& tmask,
                              const std::vector<uint32_t>& taddrs);
  void issue_tcgen05_st_async(uint32_t wid,
                              uint64_t trace_id,
                              const ThreadMask& tmask,
                              const std::vector<uint32_t>& taddrs,
                              const std::vector<uint32_t>& values);
  bool tcgen05_ld_trace_ready(uint64_t trace_id) const;
  bool tcgen05_wait_ld(uint32_t wid);
  bool tcgen05_wait_st(uint32_t wid);
  uint32_t mma_load_async_issue(uint32_t wid, uint32_t taddr, uint32_t idesc);
  uint32_t mma_store_async_issue(uint32_t wid, uint32_t taddr, uint32_t idesc);
  uint32_t wmma_async_issue(uint32_t wid);
  void async_tensor_complete(uint32_t async_id);
  bool tmem_taddr_ready_for_mma_load(uint32_t taddr, TcuTarget target, uint32_t sparse_mode) const;
  bool tmem_taddr_ready_for_mma_store(uint32_t taddr) const;
  TmemTaddrBlockReason tmem_taddr_load_block_reason(uint32_t taddr, TcuTarget target, uint32_t sparse_mode) const;
  TmemTaddrBlockReason tmem_taddr_store_block_reason(uint32_t taddr) const;
  // Compatibility wrappers for older TensorUnit code paths. The current
  // interface is TADDR-based; these names simply forward to the canonical API.
  bool tmem_handle_ready_for_mma_load(uint32_t taddr, TcuTarget target, uint32_t sparse_mode) const {
    return tmem_taddr_ready_for_mma_load(taddr, target, sparse_mode);
  }
  bool tmem_handle_ready_for_mma_store(uint32_t taddr) const {
    return tmem_taddr_ready_for_mma_store(taddr);
  }
  TmemTaddrBlockReason tmem_handle_load_block_reason(uint32_t taddr, TcuTarget target, uint32_t sparse_mode) const {
    return tmem_taddr_load_block_reason(taddr, target, sparse_mode);
  }
  TmemTaddrBlockReason tmem_handle_store_block_reason(uint32_t taddr) const {
    return tmem_taddr_store_block_reason(taddr);
  }
  bool has_inflight_tma_handle_activity() const { return false; }
  bool lookup_tmem_window(uint32_t, uint32_t, const TmemWindowPlan** window) const {
    if (window) *window = nullptr;
    return false;
  }
  bool ensure_tmem_window_bound(uint32_t, uint32_t, TcuTarget, uint32_t, bool = false) { return false; }
  bool try_acquire_tmem_window_read_port(uint32_t, uint32_t, uint32_t) { return true; }
  bool try_acquire_tmem_window_write_port(uint32_t, uint32_t, uint32_t) { return true; }
  bool try_acquire_tmem_read_port(uint32_t, uint32_t) { return true; }
  bool try_acquire_tmem_write_port(uint32_t, uint32_t) { return true; }
  bool try_acquire_tmem_read_meta_port(uint32_t, uint32_t) { return true; }
  bool tmem_read_window_packet(uint32_t taddr, uint32_t, uint32_t packet_idx, TmemPacket* packet) {
    return packet && tmem_taddr_read_bytes(taddr, packet_idx * Tmem::kPacketBytes, packet->bytes.data(), Tmem::kPacketBytes);
  }
  bool tmem_write_window_packet(uint32_t taddr, uint32_t, uint32_t packet_idx, const TmemPacket& packet) {
    return tmem_taddr_write_bytes(taddr, packet_idx * Tmem::kPacketBytes, packet.bytes.data(), Tmem::kPacketBytes);
  }
  bool tmem_read_packet(uint32_t taddr, uint32_t packet_idx, TmemPacket* packet) {
    return packet && tmem_taddr_read_bytes(taddr, packet_idx * Tmem::kPacketBytes, packet->bytes.data(), Tmem::kPacketBytes);
  }
  bool tmem_write_packet(uint32_t taddr, uint32_t packet_idx, const TmemPacket& packet) {
    return tmem_taddr_write_bytes(taddr, packet_idx * Tmem::kPacketBytes, packet.bytes.data(), Tmem::kPacketBytes);
  }
  bool tmem_read_meta_packet(uint32_t taddr, uint32_t packet_idx, TmemPacket* packet) {
    const TmemAllocation* alloc = nullptr;
    if (!packet || !tmem_lookup_allocation(taddr, &alloc) || !alloc) return false;
    return tmem_taddr_read_bytes(taddr, (alloc->meta_col_base - alloc->payload_col_base) * Tmem::kColBytes + packet_idx * Tmem::kPacketBytes, packet->bytes.data(), Tmem::kPacketBytes);
  }
  // ===== mbarrier API (LMEM-backed; full PTX §7.6 semantics) =====
  // mbar_addr is the absolute LMEM address holding the 8 B mbarrier_state_t.
  bool mbarrier_init(uint64_t mbar_addr, uint32_t count);
  void mbarrier_invalidate(uint64_t mbar_addr);
  // Returns the phase token observed at arrival (caller may use it to wait
  // on the *previous* phase, per PTX semantics).
  uint32_t mbarrier_arrive(uint64_t mbar_addr, uint32_t decrement_count = 1);
  void mbarrier_arrive_drop(uint64_t mbar_addr);
  // expect_tx and complete_tx feed the byte-counter (PTX §7.6.4).
  void mbarrier_expect_tx(uint64_t mbar_addr, uint32_t tx_bytes);
  void mbarrier_complete_tx(uint64_t mbar_addr, uint32_t tx_bytes);
  // Blocking wait: returns false if not yet ready (warp must stall and retry);
  // returns true if phase has advanced past `phase_token`.
  bool mbarrier_wait(uint32_t wid, uint64_t mbar_addr, uint32_t phase_token);
  // Non-blocking poll: returns true if ready, false otherwise.
  bool mbarrier_test_wait(uint64_t mbar_addr, uint32_t phase_token);
  // Bounded wait: behaves like mbarrier_wait but with a CModel-side timeout
  // expressed in cycles = (1 << timeout_bucket). Returns true on success,
  // false on timeout (caller must check return value).
  bool mbarrier_try_wait(uint32_t wid, uint64_t mbar_addr,
                         uint32_t phase_token, uint32_t timeout_bucket);
  uint64_t enqueue_tmem_request(const TmemRequestDesc& desc);
  bool tmem_request_granted(uint64_t tag) const;
  void consume_tmem_request_grant(uint64_t tag);
  bool tmem_query(uint32_t taddr, uint32_t* col_span, uint32_t* size_bytes) const;
  bool tmem_lookup_allocation(uint32_t taddr, const TmemAllocation** allocation) const;
  bool tmem_transfer_region(uint32_t taddr, uint32_t* col_base, uint32_t* col_span) const;
  void tmem_set_payload_ready(uint32_t taddr, bool ready);
  void tmem_set_meta_ready(uint32_t taddr, bool ready);
  void tmem_set_meta_region(uint32_t taddr, uint32_t meta_col_base, uint32_t meta_col_span);
  bool tmem_set_row_bytes(uint32_t taddr, uint32_t row_bytes);
#endif

#ifdef EXT_V_ENABLE
  VecUnit::Ptr& vec_unit() {
    return vec_unit_;
  }
#endif

  auto& trace_pool() {
    return trace_pool_;
  }

  const PerfStats& perf_stats() const;

  PerfStats& mutable_perf_stats() {
    return perf_stats_;
  }

  uint64_t current_cycle() const {
    return perf_stats_.cycles;
  }

  uint64_t startup_arg() const;
  void dump_tensor_debug_state(std::ostream& os) const;

  int get_exitcode() const;

private:

  void schedule();
  void fetch();
  void decode();
  void issue();
  void execute();
  void commit();

#ifdef EXT_TCU_ENABLE
  // tcgen05.wait::ld/st only track tcgen05.ld/st completion, not MMA.
  bool wait_for_inflight_tcgen05_ld_st(uint32_t wid, bool wait_store);
#endif

  uint32_t core_id_;
  Socket* socket_;
  const Arch& arch_;

#ifdef EXT_TCU_ENABLE
  TensorUnit::Ptr tensor_unit_;
  std::shared_ptr<TmemSystem> tmem_system_;
  Tma::Ptr tma_;
  SimPort<TensorAsyncOpCompletion> tensor_async_op_completion_in_;
  SimPort<TensorAsyncOpCompletion> tma_async_op_completion_in_;
  SimPort<TensorAsyncOpCompletion> tmem_system_async_op_completion_in_;
#endif

#ifdef EXT_V_ENABLE
  VecUnit::Ptr vec_unit_;
#endif

  Emulator emulator_;

  std::vector<IBuffer> ibuffers_;
  Scoreboard scoreboard_;
  std::vector<Operands::Ptr> operands_;
  std::vector<Dispatcher::Ptr> dispatchers_;
  std::vector<FuncUnit::Ptr> func_units_;
  LocalMem::Ptr local_mem_;
  std::vector<LocalMemSwitch::Ptr> lmem_switch_;
  std::vector<MemCoalescer::Ptr> mem_coalescers_;

  PipelineLatch fetch_latch_;
  PipelineLatch decode_latch_;

  HashTable<instr_trace_t*> pending_icache_;
  std::list<instr_trace_t*, PoolAllocator<instr_trace_t*, 64>> pending_instrs_;

  uint64_t pending_ifetches_;

  mutable PerfStats perf_stats_;

  std::vector<TraceArbiter::Ptr> commit_arbs_;

  uint32_t commit_exe_;
  std::vector<Arbiter> ibuffer_arbs_;

  PoolAllocator<instr_trace_t, 64> trace_pool_;

#ifdef EXT_TCU_ENABLE
  using TmemAllocation = vortex::TmemAllocation;

  enum class AsyncTensorOpType : uint8_t {
    TmaLoad = 0,
    TmaStore,
    TmemCopy,
    TmemShift,
    MmaLoad,
    MmaStore,
    Wmma,
  };

  // AsyncTensorOp models one Core-side asynchronous tensor transaction.
  //
  // Hardware view:
  // - issue/launch happens in the execute stage and assigns an async id
  // - the Core-side async tensor engine then advances launch latency, TMEM
  //   packet traffic and optional shift-refill traffic one cycle at a time
  // - the TensorUnit later consumes the resulting TMEM-visible state
  //
  // The fields below are grouped by: identity, launch timing, TMEM packet
  // cursors, logical TMEM regions, and staging buffers used by the async
  // engine.
  struct AsyncTensorOp {
    uint32_t async_id = 0;
    AsyncTensorOpType type = AsyncTensorOpType::TmaLoad;
    uint32_t wid = 0;
    uint32_t wgid = 0;
    uint32_t taddr = 0;
    uint32_t idesc = 0;
    uint64_t issue_cycle = 0;
    bool completed = false;
    bool committed = false;
    uint32_t barrier_id = 0;
    uint32_t payload_size_bytes = 0;
    // GAP-2 (PTX §7.6.4 cp.async.bulk.tensor.mbarrier::complete_tx::bytes):
    // when non-zero, the AsyncTensorOp's completion path calls
    // mbarrier_complete_tx(tx_bound_mbar, tx_bytes) before the standard
    // arrival/fence wakeups fire. A value of 0 means "no bytes-bound mbar".
    uint64_t tx_bound_mbar = 0;
    uint32_t tx_bytes = 0;
  };

  struct PendingTcgen05LdStAccess {
    uint32_t thread = 0;
    bool valid = false;
    uint32_t col_base = 0;
    uint32_t col_span = 0;
    uint32_t byte_offset = 0;
    uint32_t next_packet_idx = 0;
    uint32_t last_packet_idx = 0;
    uint64_t request_tag = 0;
  };

  struct PendingTcgen05LdStOp {
    uint64_t trace_id = 0;
    uint32_t wid = 0;
    bool is_store = false;
    RegOpd dst = {RegType::None, 0};
    ThreadMask tmask;
    uint64_t issue_cycle = 0;
    uint32_t next_access = 0;
    std::vector<uint32_t> taddrs;
    std::vector<Word> values;
    std::vector<PendingTcgen05LdStAccess> accesses;
  };

  // Runtime mbarrier state. The LMEM-visible 8 B object remains
  // mbarrier_state_t; this runtime state carries extra simulator bookkeeping
  // such as waiters and expected_tx_count.
  struct mbarrier_runtime_state_t {
    bool     valid = false;
    uint32_t expected_arrival_count = 0;  // [19:0]   reset target on phase advance
    uint32_t pending_arrival_count  = 0;  // [61:32]  decremented by mbarrier_arrive
    uint32_t pending_tx_count       = 0;  // [31:20]  bytes from mbarrier_complete_tx
    uint32_t expected_tx_count      = 0;  // accumulated by mbarrier_expect_tx;
                                          //  not in LMEM 8 B layout per PTX 7.6.4
                                          //  -- tracked Core-side only
    uint32_t phase                  = 0;  // [63:62]  parity bit (toggles on advance)
    WarpMask waiters_bitmap;
  };

  struct FenceWaitState {
    bool active = false;
    TcuFenceMode mode = TcuFenceMode::Before;
  };

  std::unordered_map<uint32_t, AsyncTensorOp> async_tensor_ops_;
  std::unordered_map<uint64_t, PendingTcgen05LdStOp> pending_tcgen05_ldst_ops_;
  // Same-Core-tick taddr reservations used to serialize multiple tensor-memory
  // issue attempts within one execute phase without exposing those updates as
  // globally visible cross-module state until the next published snapshot.
  std::unordered_set<uint32_t> execute_cycle_tmem_shift_reserved_taddrs_;
  std::unordered_map<uint32_t, WarpMask> async_tensor_waiters_;
  WarpMask tcgen05_ld_waiters_;
  WarpMask tcgen05_st_waiters_;
  // mbarrier state keyed by LMEM address (PTX semantics: object lives in LMEM).
  // Each mbar_addr in [LMEM_BASE_ADDR, LMEM_BASE_ADDR+16K) maps to one entry;
  // Core mirrors the entry's LMEM-visible bits to LocalMem on every update.
  std::unordered_map<uint64_t, mbarrier_runtime_state_t> mbarriers_;
  // Per-warp wait targets: mbar_addr -> phase parity the warp is waiting on.
  std::vector<std::unordered_map<uint64_t, uint32_t>> mbarrier_wait_targets_;
  std::vector<FenceWaitState> fence_wait_states_;
  uint32_t next_async_id_;
  uint64_t last_tensor_completion_drain_cycle_;
  uint64_t last_tma_completion_drain_cycle_;
  uint64_t last_tcgen05_ldst_advance_cycle_;

  void advance_async_tensor_engine();
  uint32_t launch_lmem_to_tmem_copy(uint32_t wid,
                                    uint32_t col_base,
                                    uint32_t col_span,
                                    uint64_t lmem_addr,
                                    uint32_t byte_offset,
                                    uint32_t total_bytes);
  void publish_visible_tensor_mem_state();
  void drain_tensor_execute_completion_notices();
  void drain_tma_completion_notices();
  void advance_tcgen05_ldst_async_ops();
  void on_async_tensor_op_completed(AsyncTensorOp& op);
  void resume_async_waiters(uint32_t async_id);
  WarpMask warpgroup_mask(uint32_t wgid) const;
  void try_resume_fence_waiters();
  bool has_pending_tcgen05_ldst(uint32_t wid, bool wait_store) const;
  void try_resume_tcgen05_ldst_waiters();
  bool has_pending_async_ops(uint32_t wid, bool committed_only) const;
  bool has_pending_local_tensor_ops(uint32_t wid) const;
  void try_complete_mbarrier(uint64_t mbar_addr);
  void mirror_mbarrier_to_lmem(uint64_t mbar_addr, const mbarrier_runtime_state_t& state);
  bool tmem_region_query(uint32_t col_base, uint32_t col_span, uint32_t* size_bytes) const;
  bool tmem_taddr_busy(uint32_t taddr) const;
  bool lookup_tmem_allocation(uint32_t taddr, TmemAllocation** allocation);
  bool lookup_tmem_allocation(uint32_t taddr, const TmemAllocation** allocation) const;
#endif

  friend class LsuUnit;
  friend class AluUnit;
  friend class FpuUnit;
  friend class SfuUnit;
};

} // namespace vortex
