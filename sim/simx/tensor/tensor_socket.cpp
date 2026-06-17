// TensorSocket — one TMEM shared by N OpenTensorCores + one TMA.

#include "tensor_socket.h"
#include "core.h"
#include "debug.h"
#include "tmem_module.h"
#include "tma_module.h"

namespace vortex {

TensorSocket::TensorSocket(const SimContext& ctx, const char* name,
                           const std::vector<Core*>& cores)
  : SimObject<TensorSocket>(ctx, name)
  , CacheReqPorts(1, this)
  , CacheRspPorts(1, this) {

  uint32_t N = static_cast<uint32_t>(cores.size());
  char sname[128];

  // 1. Create N OpenTensorCore instances.
  tensor_cores_.reserve(N);
  for (uint32_t i = 0; i < N; ++i) {
    snprintf(sname, sizeof(sname), "%s_otc%d", name, i);
    tensor_cores_.emplace_back(
        std::make_shared<OpenTensorCore>(ctx, sname, cores[i], OpenTensorCoreConfig{}));
  }

  // 2. Create TMEM with N+1 read/write ports (N tensorcores + 1 TMA).
  tensor::TmemConfig tmem_cfg;
  tmem_cfg.num_read_ports  = N + 1;
  tmem_cfg.num_write_ports = N + 1;
  snprintf(sname, sizeof(sname), "%s_tmem", name);
  tmem_ = std::make_unique<tensor::Tmem>(ctx, sname, tmem_cfg);

  // 3. Create TMA (uses first core for dcache access).
  snprintf(sname, sizeof(sname), "%s_tma", name);
  tma_ = std::make_unique<tensor::Tma>(ctx, sname, cores.front(), tensor::TmaConfig{});

  uint32_t total_clients = N + 1;  // tensorcores + TMA
  uint32_t nrp = tmem_cfg.num_read_ports;
  uint32_t nwp = tmem_cfg.num_write_ports;

  // 4. TxRxArbiter: N tensorcores + TMA → TMEM read ports (M-to-1 with RoundRobin).
  {
    snprintf(sname, sizeof(sname), "%s_read_arb", name);
    auto read_arb = std::make_shared<TxRxArbiter<TensorMemPortReq, TensorMemPortRsp>>(
        ctx, sname, ArbiterType::RoundRobin, total_clients, nrp, 1, 1);
    for (uint32_t i = 0; i < N; ++i) {
      tensor_cores_[i]->TmemReadReq.bind(&read_arb->ReqIn.at(i));
      read_arb->RspIn.at(i).bind(&tensor_cores_[i]->TmemReadRsp);
    }
    tma_->TmemReadReq.bind(&read_arb->ReqIn.at(N));
    read_arb->RspIn.at(N).bind(&tma_->TmemReadRsp);
    for (uint32_t p = 0; p < nrp; ++p) {
      read_arb->ReqOut.at(p).bind(&tmem_->ReadReqPorts.at(p));
      tmem_->ReadRspPorts.at(p).bind(&read_arb->RspOut.at(p));
    }
    read_arb_ = read_arb;
  }

  // 5. TxRxArbiter: N tensorcores + TMA → TMEM write ports.
  {
    snprintf(sname, sizeof(sname), "%s_write_arb", name);
    auto write_arb = std::make_shared<TxRxArbiter<TensorMemPortReq, TensorMemPortRsp>>(
        ctx, sname, ArbiterType::RoundRobin, total_clients, nwp, 1, 1);
    for (uint32_t i = 0; i < N; ++i) {
      tensor_cores_[i]->TmemWriteReq.bind(&write_arb->ReqIn.at(i));
      write_arb->RspIn.at(i).bind(&tensor_cores_[i]->TmemWriteRsp);
    }
    tma_->TmemWriteReq.bind(&write_arb->ReqIn.at(N));
    write_arb->RspIn.at(N).bind(&tma_->TmemWriteRsp);
    for (uint32_t p = 0; p < nwp; ++p) {
      write_arb->ReqOut.at(p).bind(&tmem_->WriteReqPorts.at(p));
      tmem_->WriteRspPorts.at(p).bind(&write_arb->RspOut.at(p));
    }
    write_arb_ = write_arb;
  }

  // 6. TMA ↔ TCache (through TensorSocket's cache ports).
  tma_->CacheReq.bind(&this->CacheReqPorts.at(0));
  this->CacheRspPorts.at(0).bind(&tma_->CacheRsp);

  DT(2, "TensorSocket: " << N << " tensorcores + TMEM("
     << tmem_cfg.num_read_ports << "R/" << tmem_cfg.num_write_ports
     << "W ports) + TMA");

  this->reset();
}

TensorSocket::~TensorSocket() = default;

void TensorSocket::reset() {
  for (auto& tc : tensor_cores_) tc->reset();
  if (tmem_) tmem_->reset();
  if (tma_)  tma_->reset();
  if (read_arb_)  read_arb_->reset();
  if (write_arb_) write_arb_->reset();
}

void TensorSocket::tick() {
  if (read_arb_)  read_arb_->tick();
  if (write_arb_) write_arb_->tick();
  if (tma_)  tma_->tick();
  if (tmem_) tmem_->tick();
  for (auto& tc : tensor_cores_) tc->tick();
}

}  // namespace vortex
