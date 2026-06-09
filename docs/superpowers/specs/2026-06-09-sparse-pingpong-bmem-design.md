# Sparse Ping-Pong BMem Design

## Overview

Implement 2:4 and 1:4 sparsity support in OpenTensorCore by extending the effective K dimension (2× for 2:4, 4× for 1:4) while keeping BMem at its current 16×16 FP9 capacity. A dual-buffer (ping-pong) BMem allows loading the next K chunk while the current chunk is being computed, overlapping LMEM reads with TensorCore compute.

## Motivation

- **2:4 sparsity**: K_effective = 32. B matrix n16k32 split into 2 chunks of n16k16.
- **1:4 sparsity**: K_effective = 64. B matrix n16k64 split into 4 chunks of n16k16.
- **A matrix**: 16×16 compact (the sparse-encoded A), stored once, reused across all K rounds — AMem unchanged.
- **BMem**: Only holds one n16k16 chunk (4 lines × 8×8 FP9). Without ping-pong, compute and B-load would be serialized.

## Architecture

### Data Flow

```
Stage2a (OperandFetch)                        Stage3 (ComputePipeline)
──────────────────────────────────────────────────────────────────────────
FETCH_A→FETCH_META→FETCH_C→FETCH_B₀
CONVERT → push SOP (A+C+meta+B₀)  ────────→ FILL: A→AMem, C→CMem, B₀→bmem[0]
FETCH_B₁ (LmemReadReq)                      
CONVERT_B → push MID (B₁)         ────────→ B₁→bmem[1], swap → COMPUTE round 1
FETCH_B₂
CONVERT_B → push MID (B₂)         ────────→ B₂→bmem[0], swap → COMPUTE round 2
...
CONVERT_B → push EOP (B_last)     ────────→ B_last→bmem[idle], swap → COMPUTE last round → STORE
```

### Key Insight: Pipeline-Level Overlap

Stage3 does NOT wait for a K round to fully retire before swapping. It swaps as soon as all primitives for the current round have been **pushed into TensorCoreTop** (copied to staging registers). At that point, BMem is no longer needed for the in-flight primitives — they carry their own A/B/C copies through `tc_mul_add` pipeline. The idle BMem bank can be filled by the next chunk arriving from Stage2a while previous primitives are still flowing through the pipeline and retiring.

### Backpressure Chain

```
bmem[idle] still valid (previous chunk not yet consumed)
  → Stage3 does not pop Input
    → SimPort (Stage2a::Output → Stage3::Input) fills up
      → Stage2a Output.push blocks in CONVERT_B
        → Stage2a stops issuing LMEM read requests
          → Natural flow control, no explicit coordination needed
```

## Files Changed

| File | Change |
|---|---|
| `otc_types.h` | Add `sop`, `eop`, `b_chunk_idx`, `total_b_chunks` to `ConvertedTile` |
| `stage2_operandfetch.h` | Add `FETCH_B_CHUNK`, `CONVERT_B` states; chunk tracking fields |
| `stage2_operandfetch.cpp` | Multi-segment streaming B fetch logic |
| `stage3_compute.h` | Dual `BMem bmem_[2]`; `active_bmem_`, `b_chunk_recv`, `round_issued_subtiles` |
| `stage3_compute.cpp` | Per-round swap logic; receive chunks during COMPUTE; new accum phase model |
| `bmem.h` | **No changes** |

## Detailed Design

### 1. ConvertedTile — Stream Control Fields

```cpp
struct ConvertedTile {
  // ... existing fields unchanged ...
  uint8_t sop = 0;              // 1: start-of-packet (carries A, C, meta, B₀)
  uint8_t eop = 0;              // 1: end-of-packet (last B chunk)
  uint8_t b_chunk_idx  = 0;     // 0-based chunk index within the MMA job
  uint8_t total_b_chunks = 1;   // dense=1, 2:4=2, 1:4=4
};
```

Fragment types:
- **SOP fragment** (`sop=1`): carries `a_fp9[16][16]`, `b_fp9[16][16]` (B₀), `c_fp22[16][16]`, `sparse_meta`, all format/shape fields.
- **MID fragment** (`sop=0, eop=0`): carries only `b_fp9[16][16]` (B₁, B₂, ...).
- **EOP fragment** (`eop=1`): same as MID but marks the last chunk. May also be SOP (`sop=1, eop=1`) for dense mode.

### 2. OperandFetch — Multi-Segment Streaming

`PendingFetch` gains:
```
uint8_t  total_b_chunks;       // from sparsity_kind: none→1, 2:4→2, 1:4→4
uint8_t  b_chunk_idx;          // current chunk being loaded
uint64_t b_chunk_base;         // B₀ LMEM start address
uint32_t b_chunk_bytes;        // bytes per chunk = 16×16×elem_bytes
```

New states:
- **FETCH_B_CHUNK**: Issue LMEM reads for the next B chunk. Address = `b_chunk_base + b_chunk_idx × b_chunk_bytes`. Same per-packet read logic as existing FETCH_B, only the LMEM address advances.
- **CONVERT_B**: Convert the raw B chunk packets to `b_fp9[16][16]`, assemble a ConvertedTile with appropriate `sop`/`eop`/`b_chunk_idx` flags, push to Output. A/C/sparse_meta fields are zeroed/empty.

State machine loop:
```
FETCH_B₀ → CONVERT → push SOP (sop=1)
   ↓ if b_chunk_idx + 1 < total_b_chunks
FETCH_B_CHUNK → CONVERT_B → push MID/EOP
   ↓ loop until all chunks emitted
   ↓
IDLE (PendingFetch destroyed)
```

**Dense mode** (`total_b_chunks == 1`): SOP tile has `sop=1, eop=1`. CONVERT pushes a single tile. Behavior is identical to current code.

### 3. ComputePipeline — Dual BMem

New members:
```cpp
BMem bmem_[2];
uint32_t active_bmem_ = 0;       // which bank is being read for compute
uint32_t b_chunk_recv = 0;       // chunks received so far (1 after SOP)
uint32_t total_b_chunks = 1;
uint32_t round_issued_subtiles = 0;  // primitives pushed this round (0..3)
```

Helper methods:
```cpp
void swap() { active_bmem_ ^= 1; }
uint32_t idle_bmem() const { return active_bmem_ ^ 1; }
```

#### FILL Changes

Existing FILL writes A to AMem, C to CMem, and **B₀ to bmem_[0]**. After FILL:
- `active_bmem_ = 0`
- `b_chunk_recv = 1`
- `total_b_chunks = tile->total_b_chunks`
- `round_issued_subtiles = 0`

#### COMPUTE State Logic

Each tick, execute in order:

1. **Receive next chunk** (if conditions met):
   - `!Input.empty() && !bmem_[idle_bmem()].valid() && b_chunk_recv < total_b_chunks`
   - Pop Input, fill `bmem_[idle_bmem()]` with B data from the tile. Sets idle bank valid.

2. **Swap to next round** (if conditions met):
   - `round_issued_subtiles == 4` (all primitives for current round pushed)
   - `b_chunk_recv < total_b_chunks` (more rounds remaining)
   - `bmem_[idle_bmem()].valid()` (next chunk data ready)
   - Actions: `swap()`, `b_chunk_recv++`, `round_issued_subtiles = 0`

3. **Issue primitives** (existing logic, unchanged):
   - Read from `bmem_[active_bmem_]` and AMem
   - Push to TensorCoreTop via `push_uop`
   - Each push increments `round_issued_subtiles` tracking
   - `issue_subtile` and `issue_accum_phase` continue tracking absolute position

4. **Tick pipeline + collect retired** (existing logic, unchanged):
   - `tensorcore_.tick(true)`
   - `pop_retired` → accumulate into DMem
   - Track total retired subtiles for "all done" check

5. **All rounds complete** → STORE:
   - `b_chunk_recv == total_b_chunks && round_issued_subtiles == 4 && all retirements received`

### 4. Accum Phase Model

| Mode | rounds | phases/round | total phases | `mem_k_phase(sp, accum)` |
|---|---|---|---|---|
| dense (none) | 1 | 4 (2 K×2 bubble) | 4 | `accum / 2` |
| 2:4 | 2 | 2 (2 K, no bubble) | 4 | `accum % 2` |
| 1:4 | 4 | 2 (2 K, no bubble) | 8 | `accum % 2` |

`mem_k_phase` logic stays the same — within each round it maps accum_phase to BMem line index (K-phase 0 or 1). AMem is reused identically across all rounds since A is the compact 16×16 sparse matrix.

### 5. Dense Mode Compatibility

`total_b_chunks = 1, sop = 1, eop = 1`. Stage2a sends one tile. Stage3 FILL→COMPUTE→STORE→DONE with `b_chunk_recv = 1 = total_b_chunks`. Step 1 (receive) and step 2 (swap) in COMPUTE are never reached. Behavior is identical to current code.

## Unchanged Components

- **BMem**: Zero changes.
- **AMem, CMem, DMem**: Zero changes.
- **TensorCoreTop**: Zero changes.
- **Stage3 SimPorts**: No new ports added. Existing TmemWriteReq/Rsp used only in STORE.
- **OperandFetch SimPorts**: No new ports. Existing LmemReadReq/Rsp reused for all B chunk loads.
