# Collector Buffer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement ABuf / MBuf / BBuf[4] collector buffers for TCU_WMMA operand reuse, reducing redundant TMEM/LMEM reads when A or B matrices are shared across consecutive MMA instructions.

**Architecture:** Three independent buffer classes (ABuf, MBuf, BBuf) mirroring AMem/MetaMem/BMem storage dimensions, each with CollectorState tracking ownership and compute-inflight status. Stage1 checks readiness via OpenTensorCore::collector_ready() and stalls if busy. Stage3 routes reads/writes between collector buffers and normal SRAMs based on ConvertedTile flags.

**Tech Stack:** C++17, Vortex GPU simulator framework (SimObject/SimPort), OpenTensorCore 3-stage pipeline

**Spec:** `docs/superpowers/specs/2026-06-17-collector-buffer-design.md`

---

### Task 1: Rename collector_a_state → collector_buffer

**Files:**
- Modify: `sim/simx/common/types.h:830`
- Modify: `sim/simx/tensor/open_tensorcore/otc_types.h:45`
- Modify: `sim/simx/tensor/open_tensorcore/stage1_tcdecode.cpp:187,220`
- Modify: `sim/simx/decode.cpp:1251`

- [ ] **Step 1: Rename field in types.h (IntrTcuArgs)**

```cpp
// types.h line 830 — change:
uint8_t collector_a_fill = 0;     // qualifier[5:4] (fill/use/lastuse/discard)
// to:
uint8_t collector_buffer = 0;     // qualifier[5:4] (fill/use/lastuse/discard)
```

- [ ] **Step 2: Rename field in otc_types.h (DecodedMmaJob)**

```cpp
// otc_types.h line 45 — change:
uint8_t  collector_a_state = 0;  // qualifier[5:4] — fill/use/lastuse/discard
// to:
uint8_t  collector_buffer = 0;   // qualifier[5:4] — fill/use/lastuse/discard
```

- [ ] **Step 3: Rename in stage1_tcdecode.cpp**

```cpp
// stage1_tcdecode.cpp line 187 — change:
uint8_t collector_a_state = (qualifier >> 4) & 0x3;
// to:
uint8_t collector_buffer = (qualifier >> 4) & 0x3;

// stage1_tcdecode.cpp line 220 — change:
job.collector_a_state = collector_a_state;
// to:
job.collector_buffer = collector_buffer;
```

- [ ] **Step 4: Rename comment in decode.cpp**

```cpp
// decode.cpp line 1251 — change:
//   [5:4] collector_a_state (fill/use/lastuse/discard)
// to:
//   [5:4] collector_buffer (fill/use/lastuse/discard)
```

- [ ] **Step 5: Build verify**

```bash
make -C sim/simx VORTEX_HOME=/mnt/d/wode_code_trunk/vortex_back_up/vortex_4_32_v2/vortex -j$(nproc) 2>&1 | grep -E "error:" | grep -v "decode.cpp:88\|core.cpp:266\|emulator.cpp:207"
```
Expected: no output (only 3 pre-existing errors)

- [ ] **Step 6: Commit**

```bash
git add sim/simx/common/types.h sim/simx/tensor/open_tensorcore/otc_types.h sim/simx/tensor/open_tensorcore/stage1_tcdecode.cpp sim/simx/decode.cpp
git commit -m "refactor: rename collector_a_state to collector_buffer

Matches PTX tcgen05 naming convention. 2-bit field at qualifier[5:4]
controls ABuf (ws=0) or BBuf (ws=1) fill/use/lastuse/discard semantics.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Add bbuf_idx and collector_buffer fields to DecodedMmaJob and ConvertedTile

**Files:**
- Modify: `sim/simx/tensor/open_tensorcore/otc_types.h`

- [ ] **Step 1: Add fields to DecodedMmaJob**

```cpp
// otc_types.h — add after lanes_off in DecodedMmaJob:
  int8_t   bbuf_idx = -1;  // ws=1: which BBuf[0-3]; -1 = not using BBuf
```

- [ ] **Step 2: Add fields to ConvertedTile**

```cpp
// otc_types.h — add after total_b_chunks in ConvertedTile:
  uint8_t collector_buffer = 0;  // qualifier[5:4] passed through to Stage3
  int8_t  bbuf_idx         = -1; // ws=1: which BBuf; -1 = not using BBuf
```

- [ ] **Step 3: Build verify**

```bash
make -C sim/simx VORTEX_HOME=/mnt/d/wode_code_trunk/vortex_back_up/vortex_4_32_v2/vortex -j$(nproc) 2>&1 | grep -E "error:" | grep -v "decode.cpp:88\|core.cpp:266\|emulator.cpp:207"
```
Expected: no output

- [ ] **Step 4: Commit**

```bash
git add sim/simx/tensor/open_tensorcore/otc_types.h
git commit -m "feat: add collector_buffer and bbuf_idx to DecodedMmaJob and ConvertedTile

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Add bbuf_idx to operand_block_t

**Files:**
- Modify: `sim/simx/tensor/idescriptor.h`

- [ ] **Step 1: Document bbuf_idx in reserved0**

```cpp
// idescriptor.h — change operand_block_t.reserved0 comment:
  uint16_t reserved0;       // tail-pad to 32 B; available for future fields
// to:
  uint16_t reserved0;       // [1:0]=bbuf_idx for ws=1 TCU_WMMA, [15:2]=reserved
```

- [ ] **Step 2: Build verify**

```bash
make -C sim/simx VORTEX_HOME=/mnt/d/wode_code_trunk/vortex_back_up/vortex_4_32_v2/vortex -j$(nproc) 2>&1 | grep -E "error:" | grep -v "decode.cpp:88\|core.cpp:266\|emulator.cpp:207"
```
Expected: no output

- [ ] **Step 3: Commit**

```bash
git add sim/simx/tensor/idescriptor.h
git commit -m "feat: document bbuf_idx in operand_block_t.reserved0[1:0]

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Create collector_buffer.h with ABuf, MBuf, BBuf classes

**Files:**
- Create: `sim/simx/tensor/open_tensorcore/tensor_compute/collector_buffer.h`

- [ ] **Step 1: Write collector_buffer.h**

```cpp
// collector_buffer.h — ABuf / MBuf / BBuf collector buffer classes.
//
// Each buffer mirrors the storage dimensions of its corresponding operand SRAM
// (AMem/MetaMem/BMem) and adds warp-level ownership tracking with compute-inflight
// status. The CollectorState enforces the FILL/USE/LASTUSE/DISCARD state machine
// defined in the collector buffer design spec.
//
// Storage dimensions:
//   ABuf:  4 lines × 8×8 FP9  (64 uint16_t per line), 8 banks × 8 elems
//   MBuf:  1 packet × 64B
//   BBuf:  4 instances, each 4 lines × 8×8 FP9

#pragma once

#include <array>
#include <cstdint>
#include <cstdlib>

namespace vortex {

// ============================================================================
// CollectorState — shared across ABuf, MBuf, BBuf.
// ============================================================================
struct CollectorState {
  bool     valid            = false;
  uint32_t owner_wid        = 0;
  bool     compute_inflight = false;  // owner's MMA compute still in Stage3

  void reset() {
    valid = false;
    owner_wid = 0;
    compute_inflight = false;
  }
};

// ============================================================================
// ABuf — A-matrix collector buffer (mirrors AMem dimensions).
// ============================================================================
class ABuf {
public:
  static constexpr uint32_t kDepth     = 4;
  static constexpr uint32_t kBankCount = 8;
  static constexpr uint32_t kBankElems = 8;  // 64 elems / 8 banks
  using row_t = std::array<std::array<uint16_t, kBankElems>, kBankCount>;

  ABuf() { clear(); }

  void clear() {
    for (auto& line : lines_) {
      for (auto& bank : line) bank.fill(0);
    }
    line_valid_.fill(false);
    state_.reset();
  }

  /// Write one 8×8 FP9 line (from ConvertedTile A data).
  bool write_line(uint32_t line_idx, const uint16_t data[8][8]) {
    if (line_idx >= kDepth) return false;
    auto& dst = lines_.at(line_idx);
    for (auto& bank : dst) bank.fill(0);
    for (uint32_t i = 0; i < 8; ++i)
      for (uint32_t j = 0; j < 8; ++j)
        store_elem(dst, i * 8 + j, data[i][j]);
    line_valid_.at(line_idx) = true;
    return true;
  }

  /// Read one 8×8 primitive block for TensorCore issue.
  void read_primitive(uint32_t line_idx, uint16_t out[8][8]) const {
    if (line_idx >= kDepth || !line_valid_.at(line_idx)) std::abort();
    const auto& line = lines_.at(line_idx);
    for (uint32_t i = 0; i < 8; ++i)
      for (uint32_t j = 0; j < 8; ++j)
        out[i][j] = load_elem(line, i * 8 + j);
  }

  bool all_lines_valid() const {
    for (uint32_t l = 0; l < kDepth; ++l)
      if (!line_valid_.at(l)) return false;
    return true;
  }

  // ---- Collector lifecycle ----
  void fill(uint32_t wid) {
    state_.valid = true;
    state_.owner_wid = wid;
    state_.compute_inflight = true;
  }

  void mark_compute_started() { state_.compute_inflight = true; }
  void mark_compute_done()   { state_.compute_inflight = false; }

  void invalidate() {
    state_.valid = false;
    state_.compute_inflight = false;
    // Data left intact; valid_ gate prevents stale reads.
  }

  /// Check whether this buffer is ready for the given warp + collector op.
  /// Returns false if the instruction must stall.
  bool is_ready_for(uint32_t wid, uint8_t collector_buffer) const {
    switch (collector_buffer) {
    case 0x0: // FILL
      // Can fill if buffer is INVALID, or VALID but previous owner's compute is done.
      if (!state_.valid) return true;
      return !state_.compute_inflight;  // direct overwrite if compute done
    case 0x1: // USE
    case 0x2: // LASTUSE
      // Must be VALID and owned by this warp.
      return state_.valid && state_.owner_wid == wid;
    case 0x3: // DISCARD
      // If VALID and owned by this warp and compute still inflight → stall.
      // Otherwise (INVALID, or different owner, or compute done) → proceed.
      if (state_.valid && state_.owner_wid == wid && state_.compute_inflight)
        return false;
      return true;
    default:
      return true;
    }
  }

  bool     valid() const { return state_.valid; }
  uint32_t owner() const { return state_.owner_wid; }
  bool     compute_inflight() const { return state_.compute_inflight; }

private:
  static void store_elem(row_t& row, uint32_t idx, uint16_t v) {
    row.at(idx / kBankElems).at(idx % kBankElems) = v;
  }
  static uint16_t load_elem(const row_t& row, uint32_t idx) {
    return row.at(idx / kBankElems).at(idx % kBankElems);
  }

  std::array<row_t, kDepth> lines_;
  std::array<bool, kDepth>  line_valid_;
  CollectorState            state_;
};

// ============================================================================
// MBuf — sparse metadata collector buffer (mirrors MetaMem dimensions).
// ============================================================================
class MBuf {
public:
  static constexpr uint32_t kLineBytes = 16;

  MBuf() { clear(); }

  void clear() { data_.fill(0); state_.reset(); }

  void write(const std::array<uint8_t, 64>& data) { data_ = data; }

  void read(uint8_t out[kLineBytes], uint32_t step_m, uint32_t step_k) const {
    uint32_t line_index = step_m * 2 + step_k;
    uint32_t byte_offset = line_index * kLineBytes;
    for (uint32_t i = 0; i < kLineBytes; ++i)
      out[i] = data_[byte_offset + i];
  }

  // Collector lifecycle (identical to ABuf).
  void fill(uint32_t wid)              { state_.valid = true; state_.owner_wid = wid; state_.compute_inflight = true; }
  void mark_compute_started()          { state_.compute_inflight = true; }
  void mark_compute_done()             { state_.compute_inflight = false; }
  void invalidate()                    { state_.valid = false; state_.compute_inflight = false; }
  bool is_ready_for(uint32_t wid, uint8_t cb) const { return true; }  // follows abuf; gated by ABuf
  bool valid()                  const { return state_.valid; }

private:
  std::array<uint8_t, 64> data_;
  CollectorState           state_;
};

// ============================================================================
// BBuf — B-matrix collector buffer (4 instances, mirrors BMem dimensions).
// ============================================================================
class BBuf {
public:
  static constexpr uint32_t kDepth     = 4;
  static constexpr uint32_t kBankCount = 8;
  static constexpr uint32_t kBankElems = 8;
  using row_t = std::array<std::array<uint16_t, kBankElems>, kBankCount>;

  explicit BBuf(uint8_t index = 0) : index_(index) { clear(); }

  void clear() {
    for (auto& line : lines_) {
      for (auto& bank : line) bank.fill(0);
    }
    line_valid_.fill(false);
    state_.reset();
  }

  bool write_line(uint32_t line_idx, const uint16_t data[8][8]) {
    if (line_idx >= kDepth) return false;
    auto& dst = lines_.at(line_idx);
    for (auto& bank : dst) bank.fill(0);
    for (uint32_t i = 0; i < 8; ++i)
      for (uint32_t j = 0; j < 8; ++j)
        store_elem(dst, i * 8 + j, data[i][j]);
    line_valid_.at(line_idx) = true;
    return true;
  }

  void read_primitive(uint32_t line_idx, uint16_t out[8][8]) const {
    if (line_idx >= kDepth || !line_valid_.at(line_idx)) std::abort();
    const auto& line = lines_.at(line_idx);
    for (uint32_t i = 0; i < 8; ++i)
      for (uint32_t j = 0; j < 8; ++j)
        out[i][j] = load_elem(line, i * 8 + j);
  }

  bool all_lines_valid() const {
    for (uint32_t l = 0; l < kDepth; ++l)
      if (!line_valid_.at(l)) return false;
    return true;
  }

  // Collector lifecycle (identical to ABuf).
  void fill(uint32_t wid)              { state_.valid = true; state_.owner_wid = wid; state_.compute_inflight = true; }
  void mark_compute_started()          { state_.compute_inflight = true; }
  void mark_compute_done()             { state_.compute_inflight = false; }
  void invalidate()                    { state_.valid = false; state_.compute_inflight = false; }

  bool is_ready_for(uint32_t wid, uint8_t collector_buffer) const {
    switch (collector_buffer) {
    case 0x0: // FILL
      if (!state_.valid) return true;
      return !state_.compute_inflight;
    case 0x1: // USE
    case 0x2: // LASTUSE
      return state_.valid && state_.owner_wid == wid;
    case 0x3: // DISCARD
      if (state_.valid && state_.owner_wid == wid && state_.compute_inflight)
        return false;
      return true;
    default:
      return true;
    }
  }

  bool     valid() const { return state_.valid; }
  uint32_t owner() const { return state_.owner_wid; }
  uint8_t  index() const { return index_; }

private:
  static void store_elem(row_t& row, uint32_t idx, uint16_t v) {
    row.at(idx / kBankElems).at(idx % kBankElems) = v;
  }
  static uint16_t load_elem(const row_t& row, uint32_t idx) {
    return row.at(idx / kBankElems).at(idx % kBankElems);
  }

  std::array<row_t, kDepth> lines_;
  std::array<bool, kDepth>  line_valid_;
  CollectorState            state_;
  uint8_t                   index_;
};

}  // namespace vortex
```

- [ ] **Step 2: Build verify**

```bash
make -C sim/simx VORTEX_HOME=/mnt/d/wode_code_trunk/vortex_back_up/vortex_4_32_v2/vortex -j$(nproc) 2>&1 | grep -E "error:" | grep -v "decode.cpp:88\|core.cpp:266\|emulator.cpp:207"
```
Expected: no output (header-only, no .cpp yet, so no compilation of the new code unless included)

- [ ] **Step 3: Commit**

```bash
git add sim/simx/tensor/open_tensorcore/tensor_compute/collector_buffer.h
git commit -m "feat: add collector_buffer.h with ABuf, MBuf, BBuf classes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Update Stage1 to parse bbuf_idx and add collector buffer stall check

**Files:**
- Modify: `sim/simx/tensor/open_tensorcore/stage1_tcdecode.cpp`
- Modify: `sim/simx/tensor/open_tensorcore/opentensorcore.h`
- Modify: `sim/simx/tensor/open_tensorcore/opentensorcore.cpp`

- [ ] **Step 1: Add collector_ready() declaration to opentensorcore.h**

In the public section of `OpenTensorCore`, add:

```cpp
  /// Check whether the collector buffer (abuf or bbuf) is ready for a new
  /// MMA from warp `wid` with the given collector_buffer op and ws mode.
  /// Called by Stage1 before issuing the LsuReq for operand_block_t.
  bool collector_ready(uint32_t wid, uint8_t collector_buffer,
                       uint8_t ws, int8_t bbuf_idx) const;
```

- [ ] **Step 2: Add collector_ready() implementation to opentensorcore.cpp**

```cpp
// opentensorcore.cpp — add after feed_mma_traces():

bool OpenTensorCore::collector_ready(uint32_t wid, uint8_t collector_buffer,
                                     uint8_t ws, int8_t bbuf_idx) const {
  if (ws == 0) {
    return stage3_->abuf_ready(wid, collector_buffer);
  } else {
    return stage3_->bbuf_ready(bbuf_idx, wid, collector_buffer);
  }
}
```

- [ ] **Step 3: Add abuf_ready/bbuf_ready to ComputePipeline**

In `stage3_compute.h`, add public methods:

```cpp
  /// Collector buffer readiness queries for Stage1 stall logic.
  bool abuf_ready(uint32_t wid, uint8_t collector_buffer) const {
    return abuf_.is_ready_for(wid, collector_buffer);
  }
  bool bbuf_ready(int8_t bbuf_idx, uint32_t wid, uint8_t collector_buffer) const {
    if (bbuf_idx < 0 || bbuf_idx >= 4) return true;  // no bbuf involved
    return bbuf_[bbuf_idx].is_ready_for(wid, collector_buffer);
  }
```

- [ ] **Step 4: Parse bbuf_idx in stage1_tcdecode.cpp MMA path**

After the functional `lmem_read` of `op_block` and before filling `job` fields:

```cpp
  // stage1_tcdecode.cpp — in the LMEM response handler (pending_.valid section),
  // after op_block is read:

  // Parse bbuf_idx from operand_block_t.reserved0 when ws=1.
  int8_t bbuf_idx = -1;
  if (ws == 1) {
    bbuf_idx = static_cast<int8_t>(op_block.reserved0 & 0x3);
  }

  // Fill remaining job fields including bbuf_idx.
  job.bbuf_idx = bbuf_idx;
```

Wait—`ws` is already decoded from the qualifier in the first tick (before the LsuReq). The op_block is read in the response handler. So `ws` is in `pending_.job.ws`. Let me restructure:

```cpp
  // In the pending response handler (step 2 of tick()), after lmem_read of op_block:

  auto& job = pending_.job;
  job.a_taddr   = op_block.a_taddr;
  job.d_taddr   = op_block.d_taddr;
  job.b_sdesc   = (uint64_t(op_block.b_sdesc_hi) << 32)
                | uint64_t(op_block.b_sdesc_lo);
  job.lanes_off = op_block.lanes_off;

  // Parse bbuf_idx when ws=1 (B collector buffer mode).
  if (job.ws == 1) {
    job.bbuf_idx = static_cast<int8_t>(op_block.reserved0 & 0x3);
  } else {
    job.bbuf_idx = -1;
  }

  DT(3, "TcDecodeStage: job a_taddr=0x" << std::hex << job.a_taddr
     << " d_taddr=0x" << job.d_taddr
     << " shape=" << std::dec << job.shape_m << "x" << job.shape_n
     << " collector_buffer=" << static_cast<uint32_t>(job.collector_buffer)
     << " ws=" << static_cast<uint32_t>(job.ws)
     << " bbuf_idx=" << static_cast<int>(job.bbuf_idx));

  Output.push(job, config_.decode_latency);
  pending_.valid = false;
```

- [ ] **Step 5: Add collector buffer stall check before LsuReq in Stage1**

In the tick() function, after partial decode and before `if (LmemReadReq.full()) return;`:

```cpp
  // Check collector buffer readiness before issuing LsuReq.
  // The check uses OpenTensorCore which queries ComputePipeline's buffer state.
  // NOTE: Stage1 needs access to OTC. Add a pointer/reference to TcDecodeStage.
```

**Design decision**: Instead of giving TcDecodeStage a back-pointer to OTC, pass the readiness result as a method on OTC that Stage1 calls. Add a function pointer or reference.

Simplest approach: Add `OpenTensorCore* otc_` to TcDecodeStage, set in constructor, and call:

```cpp
  // Check collector buffer readiness (stall if buffer busy).
  if (otc_ && !otc_->collector_ready(trace->wid, collector_buffer, ws, bbuf_idx)) {
    return;  // stall: don't pop Input, don't issue LsuReq, retry next tick
  }
```

But `bbuf_idx` isn't known until after the LMEM response! The op_block read is async. So the stall check must happen AFTER the op_block response, right before `Output.push()`.

Wait—let me reconsider. The stall should happen as early as possible. But bbuf_idx is in the op_block, which requires the LMEM read. There are two cases:

1. **ws=0 (abuf)**: No bbuf_idx needed. Can check at initial decode time (before LsuReq).
2. **ws=1 (bbuf)**: Need bbuf_idx from op_block. Must wait until LMEM response.

Solution: Do the stall check at two points:
- **Before LsuReq**: If ws=0, check abuf readiness. If not ready, stall (return without issue).
- **After LMEM response**: If ws=1, check bbuf[bbuf_idx] readiness. If not ready, don't push to Output, keep pending_.valid=true, retry next tick.

Actually, simpler: always check AFTER the LMEM response, right before `Output.push()`. This handles both ws=0 and ws=1 uniformly. The LMEM read adds ~few cycles anyway, and the stall condition is rare.

```cpp
  // In the pending response handler, after op_block fields are filled:

  // Check collector buffer readiness before pushing to Output.
  if (otc_ && !otc_->collector_ready(job.wid, job.collector_buffer,
                                     job.ws, job.bbuf_idx)) {
    // Buffer not ready — keep pending valid, retry next tick.
    // The fetched op_block data is already in pending_.job, so we don't
    // need to re-read LMEM.
    pending_.stall_reason = StallReason::CollectorBusy;
    return;
  }

  Output.push(job, config_.decode_latency);
  pending_.valid = false;
```

Add to TcDecodeStage:
- `OpenTensorCore* otc_` pointer (set in constructor)
- `StallReason` enum value in PendingDecode

Actually, we don't even need a separate stall_reason. Just don't clear `pending_.valid` and don't push to Output. Next tick, the pending handler will re-enter the same block, check the response map (already consumed), skip over, and call collector_ready() again.

Let me restructure the pending handler:

```cpp
  // ---- 2. Pending MMA decode: waiting for LMEM response or collector buffer ----
  if (pending_.valid) {
    // If we already have the op_block data (lmem_read done), check collector.
    if (pending_.op_block_ready) {
      if (otc_ && !otc_->collector_ready(pending_.job.wid,
                                          pending_.job.collector_buffer,
                                          pending_.job.ws,
                                          pending_.job.bbuf_idx)) {
        return;  // stall: collector buffer not ready
      }
      // All clear — push to Output.
      DT(3, ...);
      Output.push(pending_.job, config_.decode_latency);
      pending_.valid = false;
      return;
    }

    // Wait for LMEM response.
    auto it = completed_lmem_rsp_.find(pending_.pending_req_id);
    if (it != completed_lmem_rsp_.end()) {
      operand_block_t op_block;
      core_->lmem_read(&op_block, pending_.op_block_lmem_ptr, sizeof(op_block));
      completed_lmem_rsp_.erase(it);

      auto& job = pending_.job;
      job.a_taddr   = op_block.a_taddr;
      job.d_taddr   = op_block.d_taddr;
      job.b_sdesc   = (uint64_t(op_block.b_sdesc_hi) << 32)
                    | uint64_t(op_block.b_sdesc_lo);
      job.lanes_off = op_block.lanes_off;
      if (job.ws == 1) {
        job.bbuf_idx = static_cast<int8_t>(op_block.reserved0 & 0x3);
      }
      pending_.op_block_ready = true;
      // Fall through: collector check happens next tick.
    }
    return;
  }
```

Add `bool op_block_ready = false;` to `PendingDecode` struct.

- [ ] **Step 6: Update TcDecodeStage constructor to accept OTC pointer**

```cpp
// stage1_tcdecode.h — add to TcDecodeConfig or directly:
OpenTensorCore* otc_ = nullptr;  // for collector_ready() queries
```

Update constructor in both header and cpp to accept `OpenTensorCore* otc` parameter.

- [ ] **Step 7: Update opentensorcore.cpp to pass OTC pointer to TcDecodeStage**

```cpp
// opentensorcore.cpp constructor:
snprintf(sname, sizeof(sname), "%s_stage1", name);
stage1_ = std::make_unique<TcDecodeStage>(ctx, sname, core, this, config.stage1);
```

- [ ] **Step 8: Build verify**

```bash
make -C sim/simx VORTEX_HOME=/mnt/d/wode_code_trunk/vortex_back_up/vortex_4_32_v2/vortex -j$(nproc) 2>&1 | grep -E "error:" | grep -v "decode.cpp:88\|core.cpp:266\|emulator.cpp:207"
```
Expected: no output

- [ ] **Step 9: Commit**

```bash
git add sim/simx/tensor/open_tensorcore/stage1_tcdecode.h sim/simx/tensor/open_tensorcore/stage1_tcdecode.cpp sim/simx/tensor/open_tensorcore/opentensorcore.h sim/simx/tensor/open_tensorcore/opentensorcore.cpp sim/simx/tensor/open_tensorcore/stage3_compute.h
git commit -m "feat: add collector buffer stall logic in Stage1

- Parse bbuf_idx from op_block.reserved0 when ws=1
- Add collector_ready() query chain: Stage1→OTC→ComputePipeline
- Stall before Output.push() when collector buffer not ready
- Add op_block_ready flag to PendingDecode for two-phase completion

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Update Stage3 ComputePipeline with collector buffer routing

**Files:**
- Modify: `sim/simx/tensor/open_tensorcore/stage3_compute.h`
- Modify: `sim/simx/tensor/open_tensorcore/stage3_compute.cpp`

- [ ] **Step 1: Add collector buffer members to stage3_compute.h**

```cpp
// stage3_compute.h — add includes:
#include "open_tensorcore/tensor_compute/collector_buffer.h"

// Add members in private section:
  ABuf abuf_;
  MBuf mbuf_;
  BBuf bbuf_[4] = {BBuf(0), BBuf(1), BBuf(2), BBuf(3)};
```

Note: `bbuf_[4]` uses `BBuf(uint8_t)` constructor. Since BBuf has an explicit constructor, we initialize with index values 0-3.

- [ ] **Step 2: Add abuf_ready / bbuf_ready public methods**

```cpp
// stage3_compute.h — public section:
  bool abuf_ready(uint32_t wid, uint8_t collector_buffer) const {
    return abuf_.is_ready_for(wid, collector_buffer);
  }
  bool bbuf_ready(int8_t bbuf_idx, uint32_t wid, uint8_t collector_buffer) const {
    if (bbuf_idx < 0 || bbuf_idx >= 4) return true;
    return bbuf_[bbuf_idx].is_ready_for(wid, collector_buffer);
  }
```

- [ ] **Step 3: Update reset() to clear collector buffers**

```cpp
// stage3_compute.cpp — in ComputePipeline::reset():
  abuf_.clear();
  mbuf_.clear();
  for (auto& b : bbuf_) b.clear();
```

- [ ] **Step 4: Modify fill_abc to route A and B to collector buffers**

Replace the AMem/BMem fill logic in `fill_abc`:

```cpp
void ComputePipeline::fill_abc(ActiveJob& job) {
  amem_.clear();
  bmem_[0].clear();
  bmem_[1].clear();
  cmem_.clear();
  dmem_->clear();
  tensorcore_.reset();

  const auto& tile = *job.tile;
  uint8_t  cb      = tile.collector_buffer;
  int8_t   bbi     = tile.bbuf_idx;
  bool     use_abuf = (bbi < 0) && (cb != 0x3);   // ws=0, not DISCARD
  bool     use_bbuf = (bbi >= 0) && (cb != 0x3);   // ws=1, not DISCARD

  // --- A operand ---
  if (use_abuf && cb == 0x0) {  // FILL abuf: write A to collector, not AMem
    for (uint32_t line = 0; line < AMem::kDepth; ++line) {
      uint32_t k_phase = line / 2;
      uint32_t m_block = line % 2;
      uint16_t data[8][8];
      for (uint32_t i = 0; i < 8; ++i)
        for (uint32_t k = 0; k < 8; ++k)
          data[i][k] = tile.a_fp9[m_block * 8 + i][k_phase * 8 + k];
      abuf_.write_line(line, data);
    }
    abuf_.fill(job.tile->wid);
    // Also fill mbuf if sparse.
    if (tile.has_sparse_meta) {
      mbuf_.write(tile.sparse_meta);
      mbuf_.fill(job.tile->wid);
    }
    staged_has_sparse_meta_ = tile.has_sparse_meta;
    if (tile.has_sparse_meta) staged_sparse_meta_ = tile.sparse_meta;
  } else if (use_abuf && (cb == 0x1 || cb == 0x2)) {  // USE/LASTUSE: A from abuf
    // Don't write AMem; abuf already has data. Mark compute started.
    abuf_.mark_compute_started();
    if (tile.has_sparse_meta || staged_has_sparse_meta_) {
      mbuf_.mark_compute_started();
    }
  } else {
    // Normal AMem path (DISCARD or bbuf mode).
    for (uint32_t line = 0; line < AMem::kDepth; ++line) {
      uint32_t k_phase = line / 2;
      uint32_t m_block = line % 2;
      uint16_t data[8][8];
      for (uint32_t i = 0; i < 8; ++i)
        for (uint32_t k = 0; k < 8; ++k)
          data[i][k] = tile.a_fp9[m_block * 8 + i][k_phase * 8 + k];
      amem_.write_converted_line(line, data);
    }
    staged_has_sparse_meta_ = tile.has_sparse_meta;
    if (tile.has_sparse_meta) staged_sparse_meta_ = tile.sparse_meta;
  }

  // --- B operand ---
  if (use_bbuf && cb == 0x0) {  // FILL bbuf
    for (uint32_t line = 0; line < BMem::kDepth; ++line) {
      uint32_t k_phase = line / 2;
      uint32_t n_block = line % 2;
      uint16_t data[8][8];
      for (uint32_t k = 0; k < 8; ++k)
        for (uint32_t j = 0; j < 8; ++j)
          data[k][j] = tile.b_fp9[k_phase * 8 + k][n_block * 8 + j];
      bbuf_[bbi].write_line(line, data);
    }
    bbuf_[bbi].fill(job.tile->wid);
  } else if (use_bbuf && (cb == 0x1 || cb == 0x2)) {  // USE/LASTUSE: B from bbuf
    bbuf_[bbi].mark_compute_started();
  } else {
    // Normal BMem path.
    for (uint32_t line = 0; line < BMem::kDepth; ++line) {
      uint32_t k_phase = line / 2;
      uint32_t n_block = line % 2;
      uint16_t data[8][8];
      for (uint32_t k = 0; k < 8; ++k)
        for (uint32_t j = 0; j < 8; ++j)
          data[k][j] = tile.b_fp9[k_phase * 8 + k][n_block * 8 + j];
      bmem_[0].write_converted_line(line, data);
    }
  }

  // --- C operand (unchanged) ---
  if (tile.has_c) {
    for (uint32_t subtile = 0; subtile < CMem::kDepth; ++subtile) {
      uint32_t m_block = subtile / 2;
      uint32_t n_block = subtile % 2;
      uint32_t data[8][8];
      for (uint32_t i = 0; i < 8; ++i)
        for (uint32_t j = 0; j < 8; ++j)
          data[i][j] = tile.c_fp22[m_block * 8 + i][n_block * 8 + j];
      cmem_.write_converted_subtile(subtile, data);
    }
  }
}
```

- [ ] **Step 5: Modify advance_compute to read A/B from collector buffers**

```cpp
// In advance_compute(), change the AMem/BMem read lines:

  // AMem line: read from abuf if USE/FILL mode with ws=0
  if (abuf_.valid() && job.tile->bbuf_idx < 0 && job.tile->collector_buffer != 0x3) {
    abuf_.read_primitive(mk * 2 + sm, a, false);
  } else {
    amem_.read_primitive(mk * 2 + sm, a, false);
  }

  // BMem line: read from bbuf if USE/FILL mode with ws=1
  int8_t bbi = job.tile->bbuf_idx;
  if (bbi >= 0 && job.tile->collector_buffer != 0x3 && bbuf_[bbi].valid()) {
    bbuf_[bbi].read_primitive(mk * 2 + sn, b, false);
  } else {
    bmem_[job.active_bmem].read_primitive(mk * 2 + sn, b, false);
  }

  // Sparse metadata: read from mbuf if available
  if (sp_kind != vt::sparse_none) {
    if (mbuf_.valid() && job.tile->bbuf_idx < 0 && job.tile->collector_buffer != 0x3) {
      uint32_t meta_line = sm * 2 + mk;
      mbuf_.read(sparse_row_meta_raw, sm, mk);  // different read interface
      for (uint32_t row = 0; row < kPrimitiveDim; ++row) {
        sparse_row_meta[row] = vortex::sparse::row_meta_bits(sparse_row_meta_raw, row);
      }
    } else if (staged_has_sparse_meta_) {
      // existing MetaMem path
      const uint8_t* meta_base = staged_sparse_meta_.data() + meta_line * 16;
      for (uint32_t row = 0; row < kPrimitiveDim; ++row) {
        sparse_row_meta[row] = vortex::sparse::row_meta_bits(meta_base, row);
      }
    }
  }
```

Wait — the mbuf read interface is different from MetaMem. Let me use the MBuf read method directly:

```cpp
  if (mbuf_.valid() && job.tile->bbuf_idx < 0 && job.tile->collector_buffer != 0x3) {
    uint8_t meta_line_buf[16];
    mbuf_.read(meta_line_buf, sm, mk);
    for (uint32_t row = 0; row < kPrimitiveDim; ++row) {
      sparse_row_meta[row] = vortex::sparse::row_meta_bits(meta_line_buf, row);
    }
  }
```

- [ ] **Step 6: Update finish_job to handle LASTUSE/DISCARD invalidate**

```cpp
// In finish_job(), add before Output.push():

  // Handle collector buffer LASTUSE/DISCARD cleanup.
  const auto& tile = *job.tile;
  if (tile.collector_buffer == 0x2 || tile.collector_buffer == 0x3) {
    if (tile.bbuf_idx < 0) {  // ws=0: invalidate abuf + mbuf
      abuf_.invalidate();
      mbuf_.invalidate();
    } else {  // ws=1: invalidate bbuf
      bbuf_[tile.bbuf_idx].invalidate();
    }
  } else {
    // For FILL/USE, mark compute done (no longer inflight).
    if (tile.bbuf_idx < 0) {
      abuf_.mark_compute_done();
      mbuf_.mark_compute_done();
    } else {
      bbuf_[tile.bbuf_idx].mark_compute_done();
    }
  }
```

- [ ] **Step 7: Build verify**

```bash
make -C sim/simx VORTEX_HOME=/mnt/d/wode_code_trunk/vortex_back_up/vortex_4_32_v2/vortex -j$(nproc) 2>&1 | grep -E "error:" | grep -v "decode.cpp:88\|core.cpp:266\|emulator.cpp:207"
```
Expected: no output

- [ ] **Step 8: Commit**

```bash
git add sim/simx/tensor/open_tensorcore/stage3_compute.h sim/simx/tensor/open_tensorcore/stage3_compute.cpp
git commit -m "feat: route A/B reads through collector buffers in Stage3

- fill_abc: write A→abuf (FILL) or skip AMem (USE/LASTUSE)
- fill_abc: write B→bbuf[i] (FILL) or skip BMem (USE/LASTUSE)
- advance_compute: read A from abuf when valid, else AMem
- advance_compute: read B from bbuf[i] when valid, else BMem
- finish_job: invalidate on LASTUSE/DISCARD, mark_compute_done otherwise

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Update Stage2a to skip FETCH_A/META/B for USE/LASTUSE

**Files:**
- Modify: `sim/simx/tensor/open_tensorcore/stage2_operandfetch.cpp`

- [ ] **Step 1: Skip FETCH_A and FETCH_META when ws=0, USE/LASTUSE**

In `start_fetch()`, after initializing PendingFetch, check collector_buffer:

```cpp
void OperandFetchStage::start_fetch(const DecodedMmaJob& job) {
  PendingFetch f;
  f.job = job;

  uint8_t  cb  = job.collector_buffer;
  int8_t   bbi = job.bbuf_idx;
  bool     skip_a   = (bbi < 0) && (cb == 0x1 || cb == 0x2);  // ws=0 USE/LASTUSE
  bool     skip_meta = skip_a;
  bool     skip_b   = (bbi >= 0) && (cb == 0x1 || cb == 0x2);  // ws=1 USE/LASTUSE

  if (!skip_a) {
    f.a_packet_count = tmem_packet_count(job.fmt_a, false);
    f.a_packets.resize(f.a_packet_count);
  } else {
    f.tmem_phase = skip_meta ? PendingFetch::TmemPhase::C    // skip A+META, go to C
                   : PendingFetch::TmemPhase::DONE;           // should not happen
    // ...actually, if we skip A and META, we need to handle the phase transitions.
  }

  // ...
```

Wait, the `start_fetch` initializes `tmem_phase = TmemPhase::A`. If we skip A+META, we need to start at C (if need_c) or DONE. This is more nuanced.

Better approach: Keep `tmem_phase = TmemPhase::A` as default, but have `try_issue_next_tmem` skip over A and META phases immediately when `skip_a` is true.

Add a `skip_tmem_a` flag to PendingFetch:

```cpp
// In PendingFetch struct:
bool skip_tmem_a = false;    // ws=0 USE/LASTUSE: skip FETCH_A and FETCH_META
bool skip_lmem_b = false;    // ws=1 USE/LASTUSE: skip FETCH_B
```

Set in `start_fetch()`:

```cpp
  uint8_t  cb  = job.collector_buffer;
  int8_t   bbi = job.bbuf_idx;
  f.skip_tmem_a = (bbi < 0) && (cb == 0x1 || cb == 0x2);  // ws=0 USE/LASTUSE
  f.skip_lmem_b = (bbi >= 0) && (cb == 0x1 || cb == 0x2);  // ws=1 USE/LASTUSE
```

Modify `try_issue_next_tmem()`:

```cpp
bool OperandFetchStage::try_issue_next_tmem(PendingFetch& f) {
  switch (f.tmem_phase) {
  case PendingFetch::TmemPhase::A: {
    if (f.skip_tmem_a) {
      // Skip A → jump to META (if needed) or C (if needed) or DONE.
      f.tmem_phase = f.need_meta ? PendingFetch::TmemPhase::META
                   : f.need_c   ? PendingFetch::TmemPhase::C
                   :              PendingFetch::TmemPhase::DONE;
      // If jumping to META but meta should also be skipped:
      if (f.skip_tmem_a && f.tmem_phase == PendingFetch::TmemPhase::META) {
        f.tmem_phase = f.need_c ? PendingFetch::TmemPhase::C
                               : PendingFetch::TmemPhase::DONE;
      }
      return (f.tmem_phase != PendingFetch::TmemPhase::DONE);
    }
    // ... existing A issue logic ...
  }
  // ... META case also needs skip check ...
```

Actually, this is getting complicated with the phase jump logic inside try_issue_next_tmem. Let me simplify: handle it in `start_fetch()` by setting the initial phase correctly.

```cpp
void OperandFetchStage::start_fetch(const DecodedMmaJob& job) {
  PendingFetch f;
  f.job = job;

  uint8_t  cb  = job.collector_buffer;
  int8_t   bbi = job.bbuf_idx;
  bool     use_abuf = (bbi < 0) && (cb == 0x1 || cb == 0x2);   // ws=0 USE/LASTUSE
  bool     use_bbuf = (bbi >= 0) && (cb == 0x1 || cb == 0x2);   // ws=1 USE/LASTUSE

  // TMEM path: skip A+META for abuf USE/LASTUSE.
  if (use_abuf) {
    f.a_packet_count = 0;
    f.tmem_phase = f.need_c ? PendingFetch::TmemPhase::C
                            : PendingFetch::TmemPhase::DONE;
  } else {
    f.a_packet_count = tmem_packet_count(job.fmt_a, false);
    f.a_packets.resize(f.a_packet_count);
  }

  f.need_meta = (job.sparsity_kind != vt::sparse_none) && !use_abuf;
  f.need_c    = (job.enable_input_d != 0);
  if (f.need_c) {
    f.c_packet_count = tmem_packet_count(job.fmt_c, true);
    f.c_packets.resize(f.c_packet_count);
  }

  // LMEM path: skip B for bbuf USE/LASTUSE.
  if (use_bbuf) {
    f.b_packet_count = 0;
    f.lmem_phase = PendingFetch::LmemPhase::DONE;
  } else {
    f.b_packet_count = tmem_packet_count(job.fmt_b, false);
    f.b_packets.resize(f.b_packet_count);
    f.b_lmem_base = sdesc_to_lmem_addr(job.b_sdesc);
  }

  // ... rest of start_fetch (chunk setup etc.) ...
```

This is clean: we set the initial phase to skip the A/META or B states entirely, and `try_issue_next_tmem/lmem` will see no packets to issue and immediately transition to DONE.

- [ ] **Step 2: Pass collector_buffer and bbuf_idx to ConvertedTile**

In `convert_and_push()` / `build_converted_tile()`:

```cpp
  tile->collector_buffer = f.job.collector_buffer;
  tile->bbuf_idx         = f.job.bbuf_idx;
```

- [ ] **Step 3: Build verify**

```bash
make -C sim/simx VORTEX_HOME=/mnt/d/wode_code_trunk/vortex_back_up/vortex_4_32_v2/vortex -j$(nproc) 2>&1 | grep -E "error:" | grep -v "decode.cpp:88\|core.cpp:266\|emulator.cpp:207"
```
Expected: no output

- [ ] **Step 4: Commit**

```bash
git add sim/simx/tensor/open_tensorcore/stage2_operandfetch.cpp sim/simx/tensor/open_tensorcore/stage2_operandfetch.h
git commit -m "feat: skip TMEM/LMEM reads for collector buffer USE/LASTUSE

- ws=0 USE/LASTUSE: skip FETCH_A and FETCH_META, start at FETCH_C
- ws=1 USE/LASTUSE: skip FETCH_B, start at DONE
- Pass collector_buffer and bbuf_idx to ConvertedTile

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Final build verify and integration test

**Files:** (all modified)

- [ ] **Step 1: Full build**

```bash
make -C sim/simx VORTEX_HOME=/mnt/d/wode_code_trunk/vortex_back_up/vortex_4_32_v2/vortex -j$(nproc) 2>&1 | grep -E "error:" | grep -v "decode.cpp:88\|core.cpp:266\|emulator.cpp:207"
```
Expected: no output

- [ ] **Step 2: Verify all object files exist**

```bash
ls -la sim/simx/obj/tensor/open_tensorcore/{stage1_tcdecode,stage2_operandfetch,stage3_compute,opentensorcore}.o
```
Expected: all files present

- [ ] **Step 3: Commit final changes**

```bash
git add -A
git diff --cached --stat
git commit -m "feat: complete collector buffer implementation

Implements ABuf/MBuf/BBuf[4] collector buffers for TCU_WMMA operand reuse:
- New collector_buffer.h with ABuf, MBuf, BBuf classes
- Stage1: bbuf_idx parsing, collector buffer stall check
- Stage2a: skip FETCH_A/META/B for USE/LASTUSE
- Stage3: route A/B reads through collector buffers
- Rename collector_a_state → collector_buffer

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
