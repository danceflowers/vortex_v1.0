// TensorSocket — one TMEM shared by N OpenTensorCores, plus one TMA.
//
// Mirrors the Core Socket hierarchy: N cores share one L1 cache → N tensorcores
// share one TMEM. The TMA bridges the TMEM to the tensor-cluster cache (TCache)
// for cp.async.bulk.tensor load/store operations.
//
// Constructor creates N OpenTensorCore instances + 1 TMEM + 1 TMA, then wires
// the read/write ports through RoundRobin arbiters (N→TMEM read/write ports).
//
// Each Core retrieves its OpenTensorCore via tensor_core(core_index) to bind
// the Vortex pipeline ports (TcuUnit ↔ OpenTensorCore Inputs/Outputs).

#pragma once

#include <simobject.h>
#include <memory>
#include <vector>
#include "types.h"
#include "tensor_mem_port_types.h"
#include "open_tensorcore/opentensorcore.h"

namespace vortex {

class Core;
namespace tensor { class Tmem; class Tma; }

class TensorSocket : public SimObject<TensorSocket> {
public:
  // External port to the TCache (tensor cluster cache).
  // TMA uses this to issue DRAM read/write requests.
  std::vector<SimPort<MemReq>> CacheReqPorts;
  std::vector<SimPort<MemRsp>> CacheRspPorts;

  TensorSocket(const SimContext& ctx, const char* name,
               const std::vector<Core*>& cores);

  ~TensorSocket();

  void reset();
  void tick();

  // Retrieve the OpenTensorCore for a given core index (0..N-1).
  OpenTensorCore::Ptr& tensor_core(uint32_t index) {
    return tensor_cores_.at(index);
  }

  uint32_t num_tensor_cores() const { return tensor_cores_.size(); }

  tensor::Tmem* tmem() const { return tmem_.get(); }
  tensor::Tma*  tma()  const { return tma_.get(); }

private:
  std::vector<OpenTensorCore::Ptr> tensor_cores_;
  std::unique_ptr<tensor::Tmem> tmem_;
  std::unique_ptr<tensor::Tma>    tma_;

  // M-to-1 arbiters: N tensorcores + TMA → TMEM read/write ports.
  TxRxArbiter<TensorMemPortReq, TensorMemPortRsp>::Ptr read_arb_;
  TxRxArbiter<TensorMemPortReq, TensorMemPortRsp>::Ptr write_arb_;

  // Request arbiters: N tensorcores → M TMEM ports.
  std::vector<TxArbiter<TensorMemPortReq>::Ptr> read_arbs_;
  std::vector<TxArbiter<TensorMemPortReq>::Ptr> write_arbs_;

  // Response demux buffers: per-tensorcore response queues.
  struct TmemRspRoute {
    uint32_t tc_idx;          // which tensorcore
    TensorMemPortRsp rsp;     // the response
  };
  std::deque<TmemRspRoute> pending_read_rsps_;
  std::deque<TmemRspRoute> pending_write_rsps_;
};

}  // namespace vortex
