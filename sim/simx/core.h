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
#include <unordered_map>
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

#ifdef EXT_TCU_ENABLE
  TensorUnit::Ptr& tensor_unit() {
    return tensor_unit_;
  }

  // 16 banks is enough to keep two resident fp16->fp32 tiles without overdriving shared tensor/TMA resources.
  static constexpr uint32_t kTmemNumBanks = 16;
  static constexpr uint32_t kTmemBankSize = 256;
  // Model TMA loads as off-chip bulk transfers instead of immediate next-cycle completion.
  static constexpr uint32_t kTmaLoadBaseLatency = 24;
  static constexpr uint32_t kTmaLoadBytesPerCycle = 64;
  static constexpr uint32_t kTmaTransposePenalty = 8;

  struct TmemPacket {
    std::array<uint8_t, 64> bytes;

    TmemPacket() {
      bytes.fill(0);
    }
  };

  struct TmaDescriptor {
    uint64_t addr = 0;
    uint32_t size_bytes = 0;
    uint32_t stride_bytes = 0;
    uint16_t rows = 0;
    uint16_t cols = 0;
    uint16_t elem_bytes = 0;
    uint16_t flags = 0;
    uint16_t tmem_base = 0;
    uint16_t meta_tmem_base = 0;
    uint16_t bank_span = 0;
    uint16_t meta_bank_span = 0;
    uint8_t tile_role = static_cast<uint8_t>(TcuTarget::None);
    uint8_t payload_kind = static_cast<uint8_t>(TcuPayloadKind::Dense);
    uint8_t reserved[14] = {};
  } __attribute__((packed));

  struct MmaDescriptor {
    uint32_t fmt_a = 0;
    uint32_t fmt_b = 0;
    uint32_t fmt_c = 0;
    uint8_t ws = 0;
    uint8_t sp = 0;
    uint8_t sparse_mode = 0;
    uint8_t reserved = 0;
  } __attribute__((packed));

  uint32_t tmem_alloc(uint32_t bank_span);
  bool tmem_free(uint32_t handle);
  void tmem_rel_permit();
  uint32_t tma_load(uint32_t wid, uint32_t handle, uint32_t desc_id, bool transpose_b);
  uint32_t tma_store(uint32_t wid, uint32_t handle, uint32_t desc_id);
  uint32_t tmem_shift(uint32_t wid, uint32_t handle);
  uint32_t mma_load_async_issue(uint32_t wid, uint32_t handle, uint32_t desc_id);
  uint32_t mma_store_async_issue(uint32_t wid, uint32_t handle, uint32_t desc_id);
  uint32_t wmma_async_issue(uint32_t wid);
  void async_tensor_complete(uint32_t async_id);
  uint32_t tc_commit(uint32_t wid, uint32_t barrier_id);
  bool tc_fence(uint32_t wid, TcuFenceMode mode);
  bool tc_wait(uint32_t wid);
  bool mbarrier_init(uint32_t barrier_id, uint32_t count);
  void mbarrier_arrive(uint32_t barrier_id);
  bool mbarrier_wait(uint32_t wid, uint32_t barrier_id);
  bool tma_wait(uint32_t wid, uint32_t async_id);
  bool tmem_read_packet(uint32_t handle, uint32_t packet_idx, TmemPacket* out);
  bool tmem_write_packet(uint32_t handle, uint32_t packet_idx, const TmemPacket& in);
  bool tmem_query(uint32_t handle, uint32_t* bank_span, uint32_t* size_bytes) const;
  bool read_mma_descriptor(uint32_t desc_id, MmaDescriptor* out);
  bool read_tma_descriptor(uint32_t desc_id, TmaDescriptor* out);
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

  int get_exitcode() const;

private:

  void schedule();
  void fetch();
  void decode();
  void issue();
  void execute();
  void commit();

  uint32_t core_id_;
  Socket* socket_;
  const Arch& arch_;

#ifdef EXT_TCU_ENABLE
  TensorUnit::Ptr tensor_unit_;
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
  struct TmemAllocation {
    bool valid = false;
    uint32_t start_bank = 0;
    uint32_t bank_span = 0;
    uint32_t row_bytes = 64;
  };

  enum class AsyncTensorOpType : uint8_t {
    TmaLoad = 0,
    TmaStore,
    TmemShift,
    MmaLoad,
    MmaStore,
    Wmma,
  };

  struct AsyncTensorOp {
    uint32_t async_id = 0;
    AsyncTensorOpType type = AsyncTensorOpType::TmaLoad;
    uint32_t wid = 0;
    uint32_t handle = 0;
    uint32_t descriptor_id = 0;
    uint64_t ready_cycle = 0;
    bool transpose_b = false;
    bool completed = false;
    bool committed = false;
    uint32_t barrier_id = 0;
  };

  struct MBarrierEntry {
    bool valid = false;
    bool phase_done = false;
    uint32_t phase = 0;
    uint32_t expected_arrivals = 0;
    uint32_t pending_arrivals = 0;
    uint32_t pending_tx = 0;
    WarpMask waiters_bitmap;
  };

  struct FenceWaitState {
    bool active = false;
    TcuFenceMode mode = TcuFenceMode::Before;
  };

  std::array<std::array<uint8_t, kTmemBankSize>, kTmemNumBanks> tmem_banks_;
  std::array<bool, kTmemNumBanks> tmem_bank_allocs_;
  std::unordered_map<uint32_t, TmemAllocation> tmem_allocations_;
  std::vector<TmaDescriptor> tma_desc_table_;
  std::vector<MmaDescriptor> mma_desc_table_;
  std::unordered_map<uint32_t, AsyncTensorOp> async_tensor_ops_;
  std::unordered_map<uint32_t, WarpMask> async_tensor_waiters_;
  std::vector<MBarrierEntry> mbarriers_;
  std::vector<std::unordered_map<uint32_t, uint32_t>> mbarrier_wait_targets_;
  std::vector<FenceWaitState> fence_wait_states_;
  uint32_t next_async_id_;
  bool descriptor_tables_loaded_;
  bool tmem_allocator_sealed_;
  bool realistic_tma_load_;

  static uint32_t pack_tmem_handle(uint32_t start_bank, uint32_t bank_span);
  static bool unpack_tmem_handle(uint32_t handle, uint32_t* start_bank, uint32_t* bank_span);
  bool ensure_descriptor_tables_loaded();
  uint32_t estimate_tma_load_latency(const TmaDescriptor& desc, bool transpose_b) const;
  void advance_async_tensor_ops();
  void process_async_tensor_op(AsyncTensorOp& op);
  void finalize_async_tensor_op(AsyncTensorOp& op);
  void resume_async_waiters(uint32_t async_id);
  void try_resume_fence_waiters();
  bool has_pending_async_ops(uint32_t wid, bool committed_only) const;
  bool has_pending_local_tensor_ops(uint32_t wid) const;
  void try_complete_mbarrier(uint32_t barrier_id);
  void mark_mbarrier_phase_active(uint32_t barrier_id);
  bool tmem_region_query(uint32_t start_bank, uint32_t bank_span, uint32_t* size_bytes) const;
  bool tmem_region_copy_in(uint32_t start_bank, uint32_t bank_span, const uint8_t* data, uint32_t size_bytes);
  bool tmem_region_copy_out(uint32_t start_bank, uint32_t bank_span, uint8_t* data, uint32_t size_bytes) const;
  bool tmem_region_shift_down(uint32_t start_bank, uint32_t bank_span, uint32_t row_bytes);
  bool tmem_copy_in(uint32_t handle, const uint8_t* data, uint32_t size_bytes);
  bool tmem_copy_out(uint32_t handle, uint8_t* data, uint32_t size_bytes) const;
  bool lookup_tmem_allocation(uint32_t handle, TmemAllocation** allocation);
  bool lookup_tmem_allocation(uint32_t handle, const TmemAllocation** allocation) const;
#endif

  friend class LsuUnit;
  friend class AluUnit;
  friend class FpuUnit;
  friend class SfuUnit;
};

} // namespace vortex
