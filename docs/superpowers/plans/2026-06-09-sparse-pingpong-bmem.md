# Sparse Ping-Pong BMem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement ping-pong BMem for 2:4/1:4 sparsity — Stage2a streams B chunks while Stage3 computes with dual BMem banks, overlapping LMEM reads with TensorCore compute.

**Architecture:** ConvertedTile gains `sop`/`eop`/`b_chunk_idx`/`total_b_chunks` stream-control fields. OperandFetch loops FETCH_B→CONVERT→FETCH_B_CHUNK→CONVERT_B, pushing one ConvertedTile per B chunk. ComputePipeline holds `BMem bmem_[2]`, receives chunks during COMPUTE, swaps when all primitives for the current K round are pushed into TensorCoreTop. Backpressure via SimPort: idle BMem not ready → Stage3 doesn't pop Input → Stage2a stalls.

**Tech Stack:** C++17, SimObject/SimPort framework, existing TensorCoreTop/AMem/BMem/CMem/DMem hardware models.

**Spec:** [docs/superpowers/specs/2026-06-09-sparse-pingpong-bmem-design.md](../specs/2026-06-09-sparse-pingpong-bmem-design.md)

**Files:**
| File | Action |
|---|---|
| `sim/simx/tensor/open_tensorcore/tensor_compute/fp_types.h` | Modify: add sparse constants |
| `sim/simx/tensor/open_tensorcore/otc_types.h` | Modify: add stream fields to ConvertedTile |
| `sim/simx/tensor/open_tensorcore/stage2_operandfetch.h` | Modify: add FETCH_B_CHUNK/CONVERT_B, chunk tracking |
| `sim/simx/tensor/open_tensorcore/stage2_operandfetch.cpp` | Modify: multi-segment streaming loop |
| `sim/simx/tensor/open_tensorcore/stage3_compute.h` | Modify: dual BMem, round/swap state |
| `sim/simx/tensor/open_tensorcore/stage3_compute.cpp` | Modify: per-round swap, receive-chunks-in-COMPUTE |

---

### Task 1: Define sparse constants in fp_types.h

**Files:**
- Modify: `sim/simx/tensor/open_tensorcore/tensor_compute/fp_types.h`

**Why here:** `fp_types.h` is included by `bmem.h`, `tensor_core_top.h`, `tc_mul_add.h`, `tc_select_pipe.h` — all files that use `vortex::tensor::sparse_*` constants. The constants are referenced everywhere but never defined.

- [ ] **Step 1: Add sparse mode constants at the end of fp_types.h**

After the last function (end of `fp_types.h`, before the final `#endif` if there is one — fp_types.h has no header guard convention visible, it uses `#pragma once`), append:

```cpp
// Sparse mode constants — shared across OpenTensorCore pipeline and compute library.
namespace vortex {
namespace tensor {
constexpr uint32_t sparse_none = 0;
constexpr uint32_t sparse_2_4  = 1;
constexpr uint32_t sparse_1_4  = 2;
}  // namespace tensor
}  // namespace vortex
```

Insert this after line 413 (after the `convert_c_to_fp22` function closing brace). Location in file: after `convert_c_to_fp22()`.

- [ ] **Step 2: Build to verify**

Run: `make -j$(nproc) -C sim/simx 2>&1 | tail -30`
Expected: compiles without "sparse_none not declared" errors.

- [ ] **Step 3: Commit**

```bash
git add sim/simx/tensor/open_tensorcore/tensor_compute/fp_types.h
git commit -m "feat: define sparse mode constants in fp_types.h

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Add stream-control fields to ConvertedTile

**Files:**
- Modify: `sim/simx/tensor/open_tensorcore/otc_types.h:62-96`

- [ ] **Step 1: Add fields to ConvertedTile struct**

In `otc_types.h`, inside `struct ConvertedTile`, add four fields after `d_taddr` (line 94):

```cpp
  // Stream control for sparse multi-chunk B loading.
  // sop=1: this fragment carries A, C, and sparse_meta (start of packet).
  // eop=1: this is the last B chunk (end of packet).
  // dense mode: sop=1, eop=1, total_b_chunks=1 (single tile, backward compatible).
  uint8_t sop = 0;
  uint8_t eop = 0;
  uint8_t b_chunk_idx  = 0;
  uint8_t total_b_chunks = 1;
```

The full `ConvertedTile` struct after modification ends with:

```cpp
  // TMEM address for D writeback.
  uint32_t d_taddr = 0;

  // Stream control for sparse multi-chunk B loading.
  uint8_t sop = 0;
  uint8_t eop = 0;
  uint8_t b_chunk_idx  = 0;
  uint8_t total_b_chunks = 1;
};
```

- [ ] **Step 2: Build to verify**

Run: `make -j$(nproc) -C sim/simx 2>&1 | tail -30`
Expected: compiles cleanly.

- [ ] **Step 3: Commit**

```bash
git add sim/simx/tensor/open_tensorcore/otc_types.h
git commit -m "feat: add sop/eop/b_chunk_idx/total_b_chunks to ConvertedTile

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: OperandFetch — multi-segment B streaming

**Files:**
- Modify: `sim/simx/tensor/open_tensorcore/stage2_operandfetch.h:62-133` (FetchState enum, PendingFetch struct)
- Modify: `sim/simx/tensor/open_tensorcore/stage2_operandfetch.cpp:1-388` (full file)

#### 3a: Update header — PendingFetch state and fields

- [ ] **Step 1: Add FETCH_B_CHUNK and CONVERT_B to FetchState enum**

In `stage2_operandfetch.h`, modify the `FetchState` enum (lines 62-69):

```cpp
  enum class FetchState : uint8_t {
    IDLE,
    FETCH_A,
    FETCH_META,
    FETCH_C,
    FETCH_B,
    FETCH_B_CHUNK,   // load subsequent B chunk from LMEM (non-first)
    CONVERT,
    CONVERT_B,        // convert B-only chunk and push MID/EOP tile
  };
```

- [ ] **Step 2: Add chunk tracking fields to PendingFetch struct**

In `stage2_operandfetch.h`, inside `struct PendingFetch` (after `b_lmem_base`, around line 101), add:

```cpp
    // Multi-chunk B streaming for sparse modes.
    uint8_t  total_b_chunks = 1;
    uint8_t  b_chunk_idx    = 0;
    uint64_t b_chunk_base   = 0;
    uint32_t b_chunk_bytes  = 0;
```

Also update the `reset()` method at line 104 to include these:

```cpp
    void reset() {
      state = FetchState::IDLE;
      pending_req_id = 0; pending_tag = TmemReqTag::NONE;
      a_packet_count = 0; a_packet_idx = 0; a_packets.clear();
      need_meta = false; meta_packet = {};
      need_c    = false; c_packet_count = 0; c_packet_idx = 0; c_packets.clear();
      b_packet_count = 0; b_packet_idx = 0; b_lmem_base = 0; b_packets.clear();
      total_b_chunks = 1; b_chunk_idx = 0;
      b_chunk_base = 0; b_chunk_bytes = 0;
    }
```

- [ ] **Step 3: Add helper declaration**

In the private section, after `bool read_one_b_packet(PendingFetch& f);` (line 124), add:

```cpp
  bool read_one_b_chunk_packet(PendingFetch& f);
  std::shared_ptr<ConvertedTile> build_b_only_tile(PendingFetch& f);
```

#### 3b: Update implementation — start_fetch

- [ ] **Step 4: Update start_fetch to compute chunk parameters**

In `stage2_operandfetch.cpp`, modify `start_fetch` (lines 154-181). After setting `f.b_lmem_base`, add chunk computation:

```cpp
  // Compute B chunk parameters from sparsity mode.
  uint32_t elem_bytes = 0;
  switch (job.fmt_b) {
  case vt::fp8::id:  elem_bytes = 1; break;
  case vt::fp16::id: elem_bytes = 2; break;
  default:            elem_bytes = 1; break;
  }
  f.b_chunk_bytes = 16 * 16 * elem_bytes;  // one n16k16 chunk in bytes

  f.total_b_chunks = 1;
  if (job.sparsity_kind == vt::sparse_2_4) f.total_b_chunks = 2;
  else if (job.sparsity_kind == vt::sparse_1_4) f.total_b_chunks = 4;
  f.b_chunk_idx  = 0;
  f.b_chunk_base = f.b_lmem_base;
```

The `start_fetch` function should now look like:

```cpp
void OperandFetchStage::start_fetch(const DecodedMmaJob& job) {
  PendingFetch f;
  f.job   = job;
  f.state = FetchState::FETCH_A;

  f.a_packet_count = tmem_packet_count(job.fmt_a, false);
  f.a_packets.resize(f.a_packet_count);

  f.need_meta = (job.sparsity_kind != vt::sparse_none);
  f.need_c    = (job.enable_input_d != 0);
  if (f.need_c) {
    f.c_packet_count = tmem_packet_count(job.fmt_c, true);
    f.c_packets.resize(f.c_packet_count);
  }

  f.b_packet_count = tmem_packet_count(job.fmt_b, false);
  f.b_packets.resize(f.b_packet_count);
  f.b_lmem_base = sdesc_to_lmem_addr(job.b_sdesc);

  // Compute B chunk parameters for sparse multi-chunk streaming.
  uint32_t elem_bytes = 0;
  switch (job.fmt_b) {
  case vt::fp8::id:  elem_bytes = 1; break;
  case vt::fp16::id: elem_bytes = 2; break;
  default:            elem_bytes = 1; break;
  }
  f.b_chunk_bytes = 16 * 16 * elem_bytes;

  f.total_b_chunks = 1;
  if (job.sparsity_kind == vt::sparse_2_4) f.total_b_chunks = 2;
  else if (job.sparsity_kind == vt::sparse_1_4) f.total_b_chunks = 4;
  f.b_chunk_idx  = 0;
  f.b_chunk_base = f.b_lmem_base;

  DT(3, "OperandFetchStage: start wid=" << job.wid
     << " a_taddr=0x" << std::hex << job.a_taddr
     << " d_taddr=0x" << job.d_taddr << std::dec
     << " a_pkts=" << f.a_packet_count
     << " b_pkts=" << f.b_packet_count
     << " c_pkts=" << f.c_packet_count
     << " b_chunks=" << (int)f.total_b_chunks);

  pending_.push_back(std::move(f));
}
```

#### 3c: Update implementation — advance_fetch state machine

- [ ] **Step 5: Add FETCH_B_CHUNK and CONVERT_B cases to advance_fetch**

In `advance_fetch` (lines 187-257), modify the state machine switch. The `FETCH_B` case stays the same (reads chunk 0). Add after FETCH_B:

```cpp
  case FetchState::FETCH_B_CHUNK: {
    // Reset B packet state for the next chunk.
    f.b_packet_idx  = 0;
    f.b_packet_count = tmem_packet_count(f.job.fmt_b, false);
    f.b_packets.clear();
    f.b_packets.resize(f.b_packet_count);
    if (read_one_b_chunk_packet(f)) {
      f.state = FetchState::CONVERT_B;
    }
    break;
  }
  case FetchState::CONVERT_B: {
    auto tile = build_b_only_tile(f);
    if (tile) {
      DT(3, "OperandFetchStage: B chunk " << (int)f.b_chunk_idx
         << "/" << (int)f.total_b_chunks << " wid=" << f.job.wid);
      // Push to Output; backpressure stalls if Stage3 Input is full.
      Output.push(std::move(tile), 1);
      ++f.b_chunk_idx;
      if (f.b_chunk_idx < f.total_b_chunks) {
        // Reset B packet state and go fetch the next chunk.
        f.b_packet_idx  = 0;
        f.b_packet_count = tmem_packet_count(f.job.fmt_b, false);
        f.b_packets.clear();
        f.b_packets.resize(f.b_packet_count);
        f.state = FetchState::FETCH_B_CHUNK;
      } else {
        // All B chunks done.
        f.reset();
      }
    }
    break;
  }
```

Also, modify the existing `FETCH_B` case (lines 239-243) to transition to FETCH_B_CHUNK if there are more chunks, instead of CONVERT:

```cpp
  case FetchState::FETCH_B: {
    if (read_one_b_packet(f)) {
      f.state = FetchState::CONVERT;  // first chunk always goes to CONVERT (produces SOP tile)
    }
    break;
  }
```

The FETCH_B stays the same — it produces the SOP tile through CONVERT. The difference: in CONVERT (producing the first tile), we need to set `sop=1` and chain to `FETCH_B_CHUNK → CONVERT_B` for subsequent chunks.

- [ ] **Step 6: Modify CONVERT state to chain to FETCH_B_CHUNK**

Modify the CONVERT state case (lines 245-253) so that after building the SOP tile, if there are more B chunks, the state transitions to FETCH_B_CHUNK instead of resetting:

```cpp
  case FetchState::CONVERT: {
    auto tile = build_converted_tile(f);
    if (tile) {
      DT(3, "OperandFetchStage: done SOP wid=" << f.job.wid
         << " uuid=#" << f.job.uuid
         << " b_chunks=" << (int)f.total_b_chunks);
      Output.push(std::move(tile), config_.convert_latency);
      if (f.total_b_chunks > 1) {
        // More B chunks to load via ping-pong streaming.
        f.state = FetchState::FETCH_B_CHUNK;
        f.b_chunk_idx = 1;  // B₁ is next (B₀ was in the SOP tile)
        f.b_packet_idx  = 0;
        f.b_packet_count = tmem_packet_count(f.job.fmt_b, false);
        f.b_packets.clear();
        f.b_packets.resize(f.b_packet_count);
      } else {
        f.reset();
      }
    }
    break;
  }
```

#### 3d: Implement read_one_b_chunk_packet and build_b_only_tile

- [ ] **Step 7: Implement read_one_b_chunk_packet**

Add new function after `read_one_b_packet` (line 292):

```cpp
bool OperandFetchStage::read_one_b_chunk_packet(PendingFetch& f) {
  if (f.b_packet_idx >= f.b_packet_count) return true;
  // Address = base + chunk_idx * chunk_bytes + packet_offset.
  uint64_t addr = f.b_chunk_base
                + uint64_t(f.b_chunk_idx) * f.b_chunk_bytes
                + uint64_t(f.b_packet_idx) * kPacketBytes;
  uint64_t req_id = issue_lmem_read(addr);
  if (req_id == 0) return false;
  f.pending_req_id = req_id;
  return false;
}
```

- [ ] **Step 8: Implement build_b_only_tile**

Add new function after `build_converted_tile` (line 388):

```cpp
std::shared_ptr<ConvertedTile> OperandFetchStage::build_b_only_tile(
    PendingFetch& f) {
  auto tile = std::make_shared<ConvertedTile>();

  // Identity fields.
  tile->uuid = f.job.uuid;
  tile->wid  = f.job.wid;

  // Precision and shape.
  tile->fmt_a  = f.job.fmt_a;
  tile->fmt_b  = f.job.fmt_b;
  tile->fmt_c  = f.job.fmt_c;
  tile->fmt_d  = f.job.fmt_d;
  tile->shape_m = f.job.shape_m;
  tile->shape_n = f.job.shape_n;
  tile->sparsity_kind = f.job.sparsity_kind;
  tile->d_taddr = f.job.d_taddr;

  // Stream control.
  tile->sop = 0;
  tile->eop = (f.b_chunk_idx + 1 == f.total_b_chunks) ? 1 : 0;
  tile->b_chunk_idx = f.b_chunk_idx;
  tile->total_b_chunks = f.total_b_chunks;

  // B: 4 lines → FP9 (B-only chunk, no A/C/meta).
  uint32_t ppl = BMem::packets_per_fill_line(f.job.fmt_b);
  for (uint32_t line = 0; line < BMem::kDepth; ++line) {
    std::vector<BMem::packet_t> line_pkts;
    line_pkts.reserve(ppl);
    for (uint32_t p = 0; p < ppl; ++p) {
      uint32_t idx = line * ppl + p;
      if (idx >= f.b_packets.size()) return nullptr;
      line_pkts.push_back(f.b_packets.at(idx));
    }
    uint16_t converted[8][8];
    if (!BMem::convert_fill_packets(f.job.fmt_b, line_pkts, converted))
      return nullptr;
    uint32_t k_phase = line / 2, n_block = line % 2;
    for (uint32_t k = 0; k < 8; ++k)
      for (uint32_t j = 0; j < 8; ++j)
        tile->b_fp9[k_phase * 8 + k][n_block * 8 + j] = converted[k][j];
  }

  return tile;
}
```

- [ ] **Step 9: Modify build_converted_tile to set SOP/EOP**

In `build_converted_tile` (lines 305-388), after `tile->d_taddr = f.job.d_taddr;` (line 317), add stream control:

```cpp
  // Stream control: first tile is always SOP.
  tile->sop = 1;
  tile->eop = (f.total_b_chunks == 1) ? 1 : 0;
  tile->b_chunk_idx  = 0;
  tile->total_b_chunks = f.total_b_chunks;
```

- [ ] **Step 10: Modify LMEM response handler for FETCH_B_CHUNK tag**

The LMEM response drain in `advance_fetch` (lines 189-216) uses `f.state <= FetchState::FETCH_C` to check if the pending request is TMEM, otherwise LMEM. The `FETCH_B_CHUNK` state needs to be recognized as an LMEM-read state (not TMEM). Update the condition at line 190:

```cpp
  if (f.pending_req_id != 0) {
    if (f.state <= FetchState::FETCH_C) {
      // TMEM response handling (unchanged).
      ...
    } else {
      // LMEM response handling (FETCH_B or FETCH_B_CHUNK).
      auto it = completed_lmem_rsp_.find(f.pending_req_id);
      if (it == completed_lmem_rsp_.end()) return;
      uint64_t addr = 0;
      if (f.state == FetchState::FETCH_B) {
        addr = f.b_lmem_base + f.b_packet_idx * kPacketBytes;
      } else {
        addr = f.b_chunk_base
             + uint64_t(f.b_chunk_idx) * f.b_chunk_bytes
             + uint64_t(f.b_packet_idx) * kPacketBytes;
      }
      core_->lmem_read(f.b_packets.at(f.b_packet_idx).data(), addr, kPacketBytes);
      ++f.b_packet_idx;
      completed_lmem_rsp_.erase(it);
    }
    f.pending_req_id = 0;
    f.pending_tag    = TmemReqTag::NONE;
  }
```

- [ ] **Step 11: Build to verify**

Run: `make -j$(nproc) -C sim/simx 2>&1 | tail -30`
Expected: compiles cleanly.

- [ ] **Step 12: Commit**

```bash
git add sim/simx/tensor/open_tensorcore/stage2_operandfetch.h
git add sim/simx/tensor/open_tensorcore/stage2_operandfetch.cpp
git commit -m "feat: multi-segment B streaming in OperandFetch for sparse ping-pong

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: ComputePipeline — dual BMem + per-round swap

**Files:**
- Modify: `sim/simx/tensor/open_tensorcore/stage3_compute.h:46-119` (ComputeConfig, ActiveJob, ComputePipeline)
- Modify: `sim/simx/tensor/open_tensorcore/stage3_compute.cpp:1-398` (full file)

#### 4a: Update header

- [ ] **Step 1: Increase input_depth in ComputeConfig**

In `stage3_compute.h` line 41, change `input_depth` from 1 to 2 to allow one chunk to be queued while the previous is being consumed:

```cpp
  uint32_t input_depth       = 2;  // Input port (ConvertedTile from Stage2)
                                    // 2 allows next B chunk to arrive during COMPUTE
```

- [ ] **Step 2: Add round/swap tracking to ActiveJob**

In `stage3_compute.h`, inside `ActiveJob` (after `uint32_t final_retired_subtiles = 0;`, around line 75), add:

```cpp
    // Ping-pong BMem state.
    uint32_t active_bmem  = 0;    // which BMem bank is used for compute (0 or 1)
    uint32_t b_chunk_recv = 0;    // number of B chunks received (1 after SOP)
    uint32_t total_b_chunks = 1;
    uint32_t round_issued_subtiles = 0;  // primitives pushed in current K round (0..3)
```

And add tracking for how many accumulator phases the current Sparsity mode uses (per round):

```cpp
    uint32_t phases_per_round = 2;   // num accum phases per K round (dense=4, sparse=2)
```

Update `reset()` at line 81 to include:

```cpp
    void reset() {
      tile.reset();
      state = ComputeState::IDLE;
      issue_subtile = 0; issue_accum_phase = 0;
      final_retired_subtiles = 0;
      d_store_packets.clear(); d_packet_idx = 0;
      pending_wrsp_id = 0;
      active_bmem = 0; b_chunk_recv = 0; total_b_chunks = 1;
      round_issued_subtiles = 0; phases_per_round = 2;
    }
```

- [ ] **Step 3: Change BMem to array of 2 in ComputePipeline class**

In `stage3_compute.h`, change line 109 from:

```cpp
  BMem          bmem_;
```

To:

```cpp
  BMem          bmem_[2];      // dual ping-pong banks
```

- [ ] **Step 4: Add helper methods declarations**

In the private section of ComputePipeline (after `advance_compute` declaration, around line 91), add:

```cpp
  void fill_b_bank(uint32_t bank_idx, const std::shared_ptr<ConvertedTile>& tile);
  bool try_receive_next_chunk(ActiveJob& job);
  bool try_swap_to_next_round(ActiveJob& job);
```

#### 4b: Update implementation — constructor / reset / tick

- [ ] **Step 5: Update reset() to clear both BMem banks**

In `stage3_compute.cpp`, modify `reset()` (lines 65-76):

```cpp
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
}
```

- [ ] **Step 6: Update FILL to write B₀ to bmem_[0] and set chunk tracking**

In `tick()`, modify the FILL case (lines 99-102). Replace:

```cpp
  case ComputeState::FILL:
    fill_abc(job);
    job.state = ComputeState::COMPUTE;
    break;
```

With:

```cpp
  case ComputeState::FILL:
    fill_abc(job);
    // After FILL, B₀ is in bmem_[0]. Initialize stream tracking from SOP tile.
    job.active_bmem = 0;
    job.b_chunk_recv = 1;
    job.total_b_chunks = job.tile->total_b_chunks;
    job.round_issued_subtiles = 0;
    // Determine accum phases per K round.
    if (job.tile->sparsity_kind == vt::sparse_none) {
      job.phases_per_round = kDenseAccumPhases;  // 4 (2 K-phases × 2 bubble)
    } else {
      job.phases_per_round = 2;  // sparse: 2 K-phases, no bubble
    }
    job.state = ComputeState::COMPUTE;
    break;
```

- [ ] **Step 7: Rewrite COMPUTE state to call new helper functions**

Replace the existing COMPUTE case (lines 104-111):

```cpp
  case ComputeState::COMPUTE:
    advance_compute(job);
    if (job.final_retired_subtiles == kSubtiles) {
      if (build_store_packets(job)) {
        job.state = ComputeState::STORE;
      }
    }
    break;
```

With:

```cpp
  case ComputeState::COMPUTE: {
    // Step A: try to receive the next B chunk from Stage2a.
    try_receive_next_chunk(job);

    // Step B: try to swap to the next K round.
    try_swap_to_next_round(job);

    // Step C: drive TensorCoreTop compute.
    advance_compute(job);

    // Step D: check if all K rounds and all subtiles are fully retired.
    // "All done" = all B chunks received + current round's primitives all
    // issued + all final_retired_subtiles accounted for.
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
```

#### 4c: Implement new helper functions

- [ ] **Step 8: Implement try_receive_next_chunk**

Add before `fill_abc` (around line 95):

```cpp
bool ComputePipeline::try_receive_next_chunk(ActiveJob& job) {
  // Only receive if: there are more chunks to get, idle bank is free, and data is available.
  if (job.b_chunk_recv >= job.total_b_chunks) return false;
  if (bmem_[1 - job.active_bmem].valid()) return false;  // idle bank still has old data
  if (Input.empty()) return false;

  auto tile = Input.front();

  // Validate this is the expected chunk.
  if (tile->b_chunk_idx != job.b_chunk_recv) return false;

  Input.pop();

  // Fill the idle BMem bank with just the B data from this fragment.
  fill_b_bank(1 - job.active_bmem, tile);

  DT(4, "ComputePipeline: recv B chunk " << (int)tile->b_chunk_idx
     << "/" << (int)job.total_b_chunks
     << " wid=" << job.tile->wid);
  return true;
}
```

- [ ] **Step 9: Implement try_swap_to_next_round**

Add after `try_receive_next_chunk`:

```cpp
bool ComputePipeline::try_swap_to_next_round(ActiveJob& job) {
  // Swap when: all primitives for current round are pushed into the pipeline,
  // there are more K rounds, and the idle bank has the next B chunk ready.
  if (job.round_issued_subtiles < kSubtiles) return false;
  if (job.b_chunk_recv >= job.total_b_chunks) return false;
  if (!bmem_[1 - job.active_bmem].valid()) return false;

  // Swap to the idle bank.
  job.active_bmem = 1 - job.active_bmem;
  job.b_chunk_recv++;
  job.round_issued_subtiles = 0;
  // Reset accum phase tracking for the new round.
  // issue_subtile and issue_accum_phase continue globally — but within each
  // round, the mem_k_phase resets naturally (accum_phase 0,1 → K-phase 0,1).

  DT(4, "ComputePipeline: swap to BMem bank " << job.active_bmem
     << " round " << job.b_chunk_recv << "/" << job.total_b_chunks
     << " wid=" << job.tile->wid);
  return true;
}
```

- [ ] **Step 10: Implement fill_b_bank**

Add after `try_swap_to_next_round`:

```cpp
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
```

#### 4d: Update existing functions

- [ ] **Step 11: Update fill_abc to write B₀ to bmem_[0]**

In `fill_abc` (lines 141-196), change the BMem fill section (lines 163-174). Replace `bmem_` with `bmem_[0]`:

```cpp
  // Fill BMem[0]: 4 lines, each 8×8 FP9 block.
  for (uint32_t line = 0; line < BMem::kDepth; ++line) {
    uint32_t k_phase = line / 2;
    uint32_t n_block = line % 2;
    uint16_t data[8][8];
    for (uint32_t k = 0; k < 8; ++k) {
      for (uint32_t j = 0; j < 8; ++j) {
        data[k][j] = tile.b_fp9[k_phase * 8 + k][n_block * 8 + j];
      }
    }
    bmem_[0].write_converted_line(line, data);
  }
```

- [ ] **Step 12: Update advance_compute to use active BMem and per-round issue tracking**

In `advance_compute` (lines 202-278), modify:

a) Change `bmem_.read_primitive` (line 227) to `bmem_[job.active_bmem].read_primitive`:

```cpp
    bmem_[job.active_bmem].read_primitive(mk * 2 + sn, b, false);
```

b) Update `accum_phase_count` (lines 26-31) to use `phases_per_round`:

```cpp
uint32_t ComputePipeline::accum_phase_count(const uint32_t sparsity_kind) {
  if (sparsity_kind == vt::sparse_none) {
    return kDenseAccumPhases;  // 4: 2 K-phases × 2 pipeline bubbles
  }
  return 2;  // sparse 2:4 or 1:4: 2 K-phases, no bubble
}
```

This function stays essentially the same — it returns phases per round.

c) After each `push_uop` call, increment `round_issued_subtiles`. Find the line after `tensorcore_.push_uop(a, b, c, meta, sp_kind, sparse_row_meta);` (line 251) and add tracking. The current code at lines 253-259 advances accum_phase and subtile. We need to also increment `round_issued_subtiles` when a subtile's last accum phase is pushed:

After the existing post-push code (lines 253-259), add:

```cpp
    // Track per-round issue progress for ping-pong swap decisions.
    if (accum + 1 == phases) {
      // This was the last accum phase for this subtile.
      job.round_issued_subtiles++;
    }
```

- [ ] **Step 13: Update mem_k_phase for sparse modes**

In `mem_k_phase` (lines 33-37), modify for sparse modes. For dense: `accum/2` (unchanged). For sparse: each round has 2 K-phases, `accum % 2` maps them correctly. Since `accum_phase` now resets per round (via the swap's effect on issue_accum_phase reset), the function becomes simpler:

Actually, `issue_accum_phase` does NOT reset per round — the spec says it continues globally. But `mem_k_phase` maps accum_phase → K-phase within a round. For dense: 4 phases/round, mapping = `accum/2` (0→0, 1→0, 2→1, 3→1). For sparse: 2 phases/round, mapping = `accum % 2` (0→0, 1→1).

But `issue_accum_phase` goes 0,1,2,3,0,1,2,3... across subtiles and rounds. We need it to be per-subtile. Looking at the code flow:

In `advance_compute`: for subtile 0, accum goes 0→1→2→3→(next subtile). Then subtile 1, accum goes 0→1→2→3→... This means within a subtile, accum goes through all phases.

For the K-phase selection, what matters is:
- Dense (1 round): accum 0,1 → K0; accum 2,3 → K1 ✓
- 2:4 sparse (2 rounds): per subtile, accum 0→K0, accum 1→K1. Then round 2: accum 0→K0, accum 1→K1. But the BMem is swapped so K0/K1 data are from the next chunk. ✓
- 1:4: same but 4 rounds ✓

The current `mem_k_phase` is: `(sparsity_kind == vt::sparse_none) ? (accum / 2) : accum`. This is correct for all modes when `accum` is the per-subtile accum_phase_counter. Keep it unchanged.

So `mem_k_phase` does NOT need modification. Good.

- [ ] **Step 14: Build to verify**

Run: `make -j$(nproc) -C sim/simx 2>&1 | tail -30`
Expected: compiles cleanly.

- [ ] **Step 15: Commit**

```bash
git add sim/simx/tensor/open_tensorcore/stage3_compute.h
git add sim/simx/tensor/open_tensorcore/stage3_compute.cpp
git commit -m "feat: dual BMem ping-pong with per-round swap in ComputePipeline

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: End-to-end build and verify

- [ ] **Step 1: Full clean build**

```bash
make clean -C sim/simx && make -j$(nproc) -C sim/simx 2>&1 | tail -40
```
Expected: compiles cleanly with no warnings.

- [ ] **Step 2: Run any existing tests**

```bash
# Check if there are existing tensor tests
find sim/simx -name "*test*" -type f | head -10
```
If tests exist, run them.

- [ ] **Step 3: Commit any remaining changes**

```bash
git status
git diff --stat
```

---

### Summary of Changes

| File | Lines changed | Description |
|---|---|---|
| `fp_types.h` | +5 | Define `sparse_none`/`sparse_2_4`/`sparse_1_4` constants |
| `otc_types.h` | +5 | Add `sop`/`eop`/`b_chunk_idx`/`total_b_chunks` to ConvertedTile |
| `stage2_operandfetch.h` | +15 | New states FETCH_B_CHUNK/CONVERT_B, chunk tracking fields |
| `stage2_operandfetch.cpp` | +120 | Multi-segment B streaming, `read_one_b_chunk_packet`, `build_b_only_tile` |
| `stage3_compute.h` | +15 | Dual BMem[2], round/swap state, helper declarations, input_depth=2 |
| `stage3_compute.cpp` | +100 | `try_receive_next_chunk`, `try_swap_to_next_round`, `fill_b_bank`, active BMem reads |
