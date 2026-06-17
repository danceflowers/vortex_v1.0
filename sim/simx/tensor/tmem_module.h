// TMEM — tensor memory (PTX-aligned physical model).
//
// Physical: 32 banks × 32-bit width. Banks are accessed through a crossbar
// (TxCrossBar); each read/write port independently contests for bank access
// every cycle.
//
// Logical addressing (PTX TADDR):
//   taddr = (col_byte << 16) | lane
//   lane    = row index          (M direction)
//   col_byte = byte offset within the row (N direction × elem_bytes)
//
// Physical mapping:
//   bank = (col_byte / 4) % 32
//   bank_byte = col_byte % 4
//   bank_row = lane
//
// A 512-bit (64B) packet spans 16 consecutive bank entries (16 × 4B = 64B),
// naturally covering bank[k]..bank[k+15] at the same bank_row for a dense
// list of columns.
//
// Instruction ports (Inputs/Outputs) carry instr_trace_t* for TMEM-management
// ops (TMEM_ALLOC, etc.). Currently pass-through with logging; actual allocation
// is a no-op placeholder until the memory manager is defined.

#pragma once

#include <simobject.h>
#include <array>
#include <cstdint>
#include <vector>
#include "instr_trace.h"
#include "tensor_mem_port_types.h"

namespace vortex {
namespace tensor {

struct TmemConfig {
  uint32_t num_read_ports  = 2;
  uint32_t num_write_ports = 2;
  uint32_t num_banks       = 32;   // fixed: warp width
  uint32_t bank_width      = 4;    // bytes per bank entry (32-bit)
  uint32_t num_rows        = 512;  // rows per bank (depth)
};

class Tmem : public SimObject<Tmem> {
public:
  static constexpr uint32_t kInvalidTaddr   = 0xFFFFFFFFu;
  static constexpr uint32_t kPacketBytes    = 64;
  static constexpr uint32_t kColBytes       = 512;  // max col_byte range
  static constexpr uint32_t kPacketsPerCol  = kColBytes / kPacketBytes;

  // Vortex pipeline ports (receive instr_trace_t* from TmemUnit).
  std::vector<SimPort<instr_trace_t*>> Inputs;
  std::vector<SimPort<instr_trace_t*>> Outputs;

  // Read ports (512-bit / 64B packets).
  std::vector<SimPort<TensorMemPortReq>> ReadReqPorts;
  std::vector<SimPort<TensorMemPortRsp>> ReadRspPorts;

  // Write ports.
  std::vector<SimPort<TensorMemPortReq>> WriteReqPorts;
  std::vector<SimPort<TensorMemPortRsp>> WriteRspPorts;

  Tmem(const SimContext& ctx, const char* name,
       const TmemConfig& config = TmemConfig{});

  ~Tmem();

  void reset();
  void tick();

  // ---- Allocation stubs (placeholder, no-op) ----
  uint32_t alloc(uint32_t col_span);          // always returns lane_base=0
  bool     free(uint32_t taddr);              // no-op
  bool     query(uint32_t taddr, uint32_t* col_span, uint32_t* size_bytes) const;

  // ---- Byte-level access (used during functional passthrough) ----
  bool read_bytes(uint32_t taddr, uint32_t byte_offset,
                  uint8_t* dst, uint32_t bytes) const;
  bool write_bytes(uint32_t taddr, uint32_t byte_offset,
                   const uint8_t* src, uint32_t bytes);

private:
  // Physical SRAM: banks_[bank][row] = 4 bytes.
  using BankRow = std::array<uint8_t, 4>;
  using BankMem = std::vector<BankRow>;
  std::array<BankMem, 32> banks_;

  // TxRxCrossBar: N ports → 32 banks (read) / N ports → 32 banks (write).
  TxRxCrossBar<TensorMemPortReq, TensorMemPortRsp>::Ptr read_xbar_;
  TxRxCrossBar<TensorMemPortReq, TensorMemPortRsp>::Ptr write_xbar_;

  TmemConfig config_;
};

}  // namespace tensor
}  // namespace vortex
