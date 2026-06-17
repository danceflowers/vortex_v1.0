// OpenTensorCore internal type definitions shared across Stage1/Stage2/Stage3.
// These types flow through TFifo pipelines between the three stages and are not
// exposed to the Vortex main pipeline (which uses instr_trace_t / ITraceData).

#pragma once

#include <array>
#include <cstdint>
#include <memory>

namespace vortex {

// ============================================================================
// DecodedMmaJob — Stage1 (TcDecode) output, consumed by Stage2 (OperandFetch).
//
// Stage1 reads the i_descriptor_t (from rs1), qualifier (raw funct7), and
// operand_block_t (from LMEM via rs2), then fills this struct with all fields
// Stage2 needs to build TMEM/LMEM read requests.
// ============================================================================

struct DecodedMmaJob {
  // Trace identity (carried from the original instr_trace_t).
  uint64_t uuid = 0;
  uint32_t wid  = 0;
  uint32_t cid  = 0;

  // Fields decoded from i_descriptor_t (idesc).
  uint32_t fmt_a = 0;
  uint32_t fmt_b = 0;
  uint32_t fmt_c = 0;
  uint32_t fmt_d = 0;
  uint32_t shape_m = 0;
  uint32_t shape_n = 0;
  uint8_t  sparsity_kind     = 0;  // 0=dense, 1=2:4, 2=1:4
  uint8_t  sparsity_meta_sel = 0;  // selects which 64B metadata packet accompanies A
  uint8_t  saturate          = 0;
  uint8_t  transpose_a       = 0;
  uint8_t  transpose_b       = 0;

  // Fields decoded from qualifier (raw funct7 bits).
  uint8_t  enable_input_d    = 0;  // qualifier[0]
  uint8_t  ws                = 0;  // qualifier[1]
  uint8_t  sp                = 0;  // qualifier[2]
  uint8_t  cta_group         = 0;  // qualifier[3]
  uint8_t  collector_buffer = 0;   // qualifier[5:4] — fill/use/lastuse/discard
  uint8_t  multicast         = 0;  // qualifier[6]

  // Fields loaded from operand_block_t in LMEM.
  uint32_t a_taddr   = 0;  // A matrix TMEM address
  uint32_t d_taddr   = 0;  // D output (and optional C input) TMEM address
  uint64_t b_sdesc   = 0;  // 64-bit B shared-memory descriptor (PTX §9.7.16.2)
  uint16_t lanes_off = 0;  // cta_group::2 cross-CTA lane offset
  int8_t   bbuf_idx = -1;  // ws=1: which BBuf[0-3]; -1 = not using BBuf
};

// ============================================================================
// ConvertedTile — Stage2 (OperandFetch) output, consumed by Stage3 (Compute).
//
// Stage2 reads A/C from TMEM and B from LMEM, converts everything to internal
// precision (A/B → FP9, C → FP22), and bundles the result here. Passed by
// shared_ptr to avoid copying ~1.5 KB of data through the TFifo.
// ============================================================================

struct ConvertedTile {
  // Trace identity.
  uint64_t uuid = 0;
  uint32_t wid  = 0;

  // A: 16×16 FP9 (E5M3), stored as uint16_t, row-major.
  uint16_t a_fp9[16][16] = {};

  // B: 16×16 FP9 (E5M3), stored as uint16_t, row-major.
  uint16_t b_fp9[16][16] = {};

  // C: 16×16 FP22 (E8M13), stored as uint32_t, row-major.
  // Valid only when has_c is true (enable_input_d != 0).
  uint32_t c_fp22[16][16] = {};
  bool     has_c = false;

  // Sparse metadata read alongside A from TMEM (64 B).
  // Valid only when has_sparse_meta is true (sparsity_kind != sparse_none).
  std::array<uint8_t, 64> sparse_meta = {};
  bool has_sparse_meta = false;

  // Precision and shape carried through for Stage3 compute / writeback.
  uint32_t fmt_a = 0;
  uint32_t fmt_b = 0;
  uint32_t fmt_c = 0;
  uint32_t fmt_d = 0;
  uint32_t shape_m = 0;
  uint32_t shape_n = 0;
  uint8_t  sparsity_kind = 0;

  // TMEM address for D writeback.
  uint32_t d_taddr = 0;

  // Stream control for sparse multi-chunk B loading.
  // sop=1: this fragment carries A, C, and sparse_meta (start of packet).
  // eop=1: this is the last B chunk (end of packet).
  // Dense mode: sop=1, eop=1, total_b_chunks=1 (single tile, backward compatible).
  uint8_t sop = 1;
  uint8_t eop = 1;
  uint8_t b_chunk_idx = 0;
  uint8_t total_b_chunks = 1;
  uint8_t collector_buffer = 0;  // qualifier[5:4] passed through to Stage3
  int8_t  bbuf_idx         = -1; // ws=1: which BBuf; -1 = not using BBuf
};

// ============================================================================
// CompletedMmaJob — Stage3 (Compute) output, consumed by OpenTensorCore.
//
// When Stage3 finishes computing all subtiles and writing D back to TMEM, it
// emits this struct. OpenTensorCore matches it to the pending instr_trace_t
// via uuid and pushes the trace to its Outputs port for commit.
// ============================================================================

struct CompletedMmaJob {
  uint64_t uuid = 0;
  bool success = false;
};

// LdStConfig — configurable latency for TCU_LD/ST DMem ↔ register.
struct LdStConfig {
  uint32_t port_depth = 1;   // SimPort depth
  uint32_t latency     = 4;  // total latency cycles (counter-based)
};

// LdStJob — passed from Stage1 (TcDecode) to Stage2b (DMemAccessStage).
// TCU_LD: DMem → convert → registers (is_load=true)
// TCU_ST: registers → convert → DMem (is_load=false)
struct instr_trace_t;

struct LdStJob {
  bool     is_load = true;
  uint32_t fmt     = 0;                          // precision format
  uint32_t rd      = 0;                          // dest reg base (LD only)
  instr_trace_t* trace = nullptr;               // original pipeline trace
  std::array<uint32_t, 256> st_values = {};      // source values (ST only)
};

}  // namespace vortex
