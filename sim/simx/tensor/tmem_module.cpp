// TMEM — simplified 32-bank × 32-bit physical SRAM with crossbar arbitration.

#include "tmem_module.h"
#include <cstring>
#include "debug.h"

namespace vortex {
namespace tensor {

// Forward decls for static SRAM helpers.
static bool taddr_to_bank(uint32_t taddr, uint32_t byte_offset,
                          uint32_t* bank, uint32_t* row, uint32_t* bank_byte);
static void access_sram(std::array<std::vector<std::array<uint8_t, 4>>, 32>& banks,
                        uint32_t taddr, uint32_t byte_off, uint32_t bytes,
                        uint8_t* data, bool is_write);
static void do_read_packet(std::array<std::vector<std::array<uint8_t, 4>>, 32>& banks,
                           const TensorMemPortReq& req, TmemPacket* pkt);
static void do_write_packet(std::array<std::vector<std::array<uint8_t, 4>>, 32>& banks,
                            const TensorMemPortReq& req);

// ---------------------------------------------------------------------------
// Constructor / destructor / reset / tick
// ---------------------------------------------------------------------------

Tmem::Tmem(const SimContext& ctx, const char* name, const TmemConfig& config)
  : SimObject<Tmem>(ctx, name)
  , Inputs(ISSUE_WIDTH, this)
  , Outputs(ISSUE_WIDTH, this)
  , config_(config) {

  // Allocate SRAM banks.
  uint32_t rows = config.num_rows;
  for (auto& bank : banks_) {
    bank.resize(rows);
    for (auto& row : bank) row.fill(0);
  }

  // ReqPorts are sinks the TMEM reads from (real depth). RspPorts are bind
  // sources (→ arbiters) and MUST be capacity-0 virtual links (simobject.h:91).
  for (uint32_t i = 0; i < config.num_read_ports; ++i) {
    ReadReqPorts.emplace_back(this, 2);
    ReadRspPorts.emplace_back(this, 0);
  }
  for (uint32_t i = 0; i < config.num_write_ports; ++i) {
    WriteReqPorts.emplace_back(this, 2);
    WriteRspPorts.emplace_back(this, 0);
  }

  // TxRxCrossBar: reads and writes share the same bank routing function.
  auto bank_sel = [](const TensorMemPortReq& req) -> uint32_t {
    uint32_t cb = ((req.taddr >> 16) & 0xFFFFu)
                + req.packet_idx * kPacketBytes;
    return (cb / 4) % 32;
  };
  {
    char rn[128], wn[128];
    snprintf(rn, sizeof(rn), "%s_rxbar", name);
    snprintf(wn, sizeof(wn), "%s_wxbar", name);
    read_xbar_  = std::make_shared<TxRxCrossBar<TensorMemPortReq, TensorMemPortRsp>>(
        ctx, rn, ArbiterType::RoundRobin, config.num_read_ports, config.num_banks, bank_sel, 1, 1);
    write_xbar_ = std::make_shared<TxRxCrossBar<TensorMemPortReq, TensorMemPortRsp>>(
        ctx, wn, ArbiterType::RoundRobin, config.num_write_ports, config.num_banks, bank_sel, 1, 1);
  }

  this->reset();
}

Tmem::~Tmem() = default;

void Tmem::reset() {
  for (auto& bank : banks_) {
    for (auto& row : bank) row.fill(0);
  }
}

void Tmem::tick() {
  DT(4, "Tmem: tick cycle=" << SimPlatform::instance().cycles());

  // ---- Instruction traces ----
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
    if (Inputs[iw].empty()) continue;
    auto trace = Inputs[iw].front();
    auto* td = dynamic_cast<TmemTraceData*>(trace->data.get());
    if (td == nullptr) {
      Inputs[iw].pop();
      Outputs[iw].push(trace, 3);
      continue;
    }
    Inputs[iw].pop();

    DT(3, "Tmem: instr trace=#" << trace->uuid
       << " funct3=0x" << std::hex << td->funct3 << std::dec
       << " r1=0x" << td->rs1_value << " r2=0x" << td->rs2_value);

    // Decode and execute TMEM instruction.
    switch (td->funct3) {
    case 0b001: // TMEM_REL_PERMIT (no-op placeholder)
      break;
    case 0b010: // TMEM_ALLOC — returns lane_base=0 (free allocation)
      // Register writeback is handled by execute stage via Core methods.
      break;
    case 0b011: // TMEM_DEALLOC (no-op placeholder)
      break;
    case 0b100: // TMEM_CP — shared-mem → TMEM (TODO: LMEM read → TMEM write)
      DT(3, "Tmem: CP taddr=0x" << std::hex << td->rs1_value
         << " sdesc=0x" << td->rs2_value << std::dec
         << " (TODO: LMEM read + bank write)");
      break;
    case 0b101: // TMEM_SHIFT (no-op placeholder)
      break;
    default:
      DT(1, "Tmem: unknown funct3=0x" << std::hex << td->funct3);
      break;
    }
    Outputs[iw].push(trace, 4);
  }

  // ---- Tick crossbars (handles routing + response delivery) ----
  read_xbar_->tick();
  write_xbar_->tick();

  // ---- Process requests on the bank side (ReOut → process → RspOut) ----
  for (uint32_t b = 0; b < config_.num_banks; ++b) {
    // Reads.
    if (!read_xbar_->ReqOut[b].empty()) {
      auto req = read_xbar_->ReqOut[b].front(); read_xbar_->ReqOut[b].pop();
      TmemPacket pkt;
      do_read_packet(banks_, req, &pkt);
      TensorMemPortRsp rsp;
      rsp.request_id = req.request_id; rsp.access_type = req.access_type;
      rsp.tag = req.tag;  // crossbar routes the response back by tag (source port)
      rsp.read_packet = pkt;
      read_xbar_->RspOut[b].push(rsp, 1);
    }
    // Writes.
    if (!write_xbar_->ReqOut[b].empty()) {
      auto req = write_xbar_->ReqOut[b].front(); write_xbar_->ReqOut[b].pop();
      do_write_packet(banks_, req);
      TensorMemPortRsp rsp;
      rsp.request_id = req.request_id; rsp.access_type = req.access_type;
      rsp.tag = req.tag;  // crossbar routes the response back by tag (source port)
      write_xbar_->RspOut[b].push(rsp, 1);
    }
  }

  // ---- Feed new requests + drain responses on the port side ----
  DT(4, "Tmem: tick cycle=" << SimPlatform::instance().cycles()
     << " read_rsp=" << read_xbar_->RspIn[0].size());
  for (uint32_t p = 0; p < config_.num_read_ports; ++p) {
    if (!ReadReqPorts[p].empty() && !read_xbar_->ReqIn[p].full()) {
      auto req = ReadReqPorts[p].front(); ReadReqPorts[p].pop();
      read_xbar_->ReqIn[p].push(req, 0);
    }
    if (!read_xbar_->RspIn[p].empty() && !ReadRspPorts[p].full()) {
      ReadRspPorts[p].push(read_xbar_->RspIn[p].front(), 0);
      read_xbar_->RspIn[p].pop();
    }
  }
  for (uint32_t p = 0; p < config_.num_write_ports; ++p) {
    if (!WriteReqPorts[p].empty() && !write_xbar_->ReqIn[p].full()) {
      auto req = WriteReqPorts[p].front(); WriteReqPorts[p].pop();
      DT(2, "Tmem: WriteReqPort[" << p << "] req_id=" << req.request_id
         << " taddr=0x" << std::hex << req.taddr << std::dec << " pkt=" << req.packet_idx);
      write_xbar_->ReqIn[p].push(req, 0);
    }
    if (!write_xbar_->RspIn[p].empty() && !WriteRspPorts[p].full()) {
      DT(2, "Tmem: WriteRspPort[" << p << "] req_id=" << write_xbar_->RspIn[p].front().request_id);
      WriteRspPorts[p].push(write_xbar_->RspIn[p].front(), 0);
      write_xbar_->RspIn[p].pop();
    }
  }
}

// ---------------------------------------------------------------------------
// Allocation stubs (no-op)
// ---------------------------------------------------------------------------

uint32_t Tmem::alloc(uint32_t) {
  return 0;  // always return lane_base=0 (no allocation tracking)
}

bool Tmem::free(uint32_t) {
  return true;
}

bool Tmem::query(uint32_t, uint32_t* col_span, uint32_t* size_bytes) const {
  if (col_span) *col_span = config_.num_banks;
  if (size_bytes) *size_bytes = config_.num_banks * config_.bank_width * config_.num_rows;
  return true;
}

// ---------------------------------------------------------------------------
// Unified SRAM access — all TADDR→bank mapping goes through here.
// ---------------------------------------------------------------------------

// Map (taddr, byte_offset) → (bank, row, bank_byte). byte_offset is relative to
// the taddr's col_byte, constrained to one bank-width chunk.
static bool taddr_to_bank(uint32_t taddr, uint32_t byte_offset,
                          uint32_t* bank, uint32_t* row, uint32_t* bank_byte) {
  uint32_t lane     = taddr & 0xFFFFu;
  uint32_t col_byte = (taddr >> 16) & 0xFFFFu;
  uint32_t cb       = col_byte + byte_offset;
  uint32_t bw       = 4;  // bank width
  if (bank)      *bank = (cb / bw) % 32;
  if (row)       *row  = lane + byte_offset / (32 * bw);
  if (bank_byte) *bank_byte = cb % bw;
  return true;
}

// Read/write a range of bytes from/to the bank SRAM.
static void access_sram(std::array<std::vector<std::array<uint8_t, 4>>, 32>& banks,
                        uint32_t taddr, uint32_t byte_off, uint32_t bytes,
                        uint8_t* data, bool is_write) {
  for (uint32_t i = 0; i < bytes; i += 4) {
    uint32_t bank, row, bb;
    taddr_to_bank(taddr, byte_off + i, &bank, &row, &bb);
    auto& br = banks.at(bank).at(row);
    if (is_write) {
      for (uint32_t b = 0; b < 4 && (i + b) < bytes; ++b)
        br[bb + b] = data[i + b];
    } else {
      uint32_t val = 0;
      for (uint32_t b = 0; b < 4; ++b)
        val |= static_cast<uint32_t>(br[bb + b]) << (b * 8);
      std::memcpy(data + i, &val, std::min<uint32_t>(4, bytes - i));
    }
  }
}

// ---- Packet-level read ----
static void do_read_packet(std::array<std::vector<std::array<uint8_t, 4>>, 32>& banks,
                           const TensorMemPortReq& req, TmemPacket* pkt) {
  access_sram(banks, req.taddr,
              req.packet_idx * 64, 64, pkt->bytes.data(), false);
}

// ---- Packet-level write ----
static void do_write_packet(std::array<std::vector<std::array<uint8_t, 4>>, 32>& banks,
                            const TensorMemPortReq& req) {
  access_sram(banks, req.taddr,
              req.packet_idx * 64, 64,
              const_cast<uint8_t*>(req.write_packet.bytes.data()), true);
}

// ---- Public byte access ----
bool Tmem::read_bytes(uint32_t taddr, uint32_t byte_offset,
                      uint8_t* dst, uint32_t bytes) const {
  access_sram(const_cast<decltype(banks_)&>(banks_),
              taddr, byte_offset, bytes, dst, false);
  return true;
}

bool Tmem::write_bytes(uint32_t taddr, uint32_t byte_offset,
                       const uint8_t* src, uint32_t bytes) {
  access_sram(banks_, taddr, byte_offset, bytes,
              const_cast<uint8_t*>(src), true);
  return true;
}

}  // namespace tensor
}  // namespace vortex
