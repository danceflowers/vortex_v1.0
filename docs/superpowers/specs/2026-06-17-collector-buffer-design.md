# Collector Buffer Design — ABuf / MBuf / BBuf[4]

Date: 2026-06-17

## 1. Motivation

当前 OpenTensorCore pipeline 中，每条 TCU_WMMA 指令都会从 TMEM 读取 A、从 LMEM 读取 B、从 TMEM 读取 C。当同一个 A（或 B）矩阵在多个连续 MMA 指令中复用时，TMEM/LMEM 读取是重复且浪费的。

PTX tcgen05 引入 collector buffer 解决此问题：一条指令将操作数存入 buffer（FILL），后续指令直接从 buffer 读取（USE），省去重复的 TMEM/LMEM 读取延迟。

## 2. Collector Buffer 编码

### 2.1 字段重命名

`collector_a_state` → `collector_buffer`，含义不变（2-bit, qualifier[5:4]）。

### 2.2 编码表

| collector_buffer | 名称 | 含义 |
|---|---|---|
| 00 | FILL | 操作数写入 collector buffer，不写 AMem/BMem；计算从 buffer 读 |
| 01 | USE | 复用 buffer 中的操作数，不从 TMEM/LMEM 读 |
| 10 | LASTUSE | 同 USE，但计算完成后 buffer 置为 INVALID |
| 11 | DISCARD | 操作数走 AMem/BMem 正常路径，计算完成后 buffer 置为 INVALID |

### 2.3 ws 语义

- **ws=0**：`collector_buffer` 控制 abuf（A 矩阵） + mbuf（稀疏 metadata）
- **ws=1**：`collector_buffer` 控制 bbuf（B 矩阵）

一个 TCU_WMMA 同一时刻只能控制 A buffer 或 B buffer，不可同时。

### 2.4 bbuf 索引

bbuf 有 4 个实例。ws=1 时，额外从 `operand_block_t.reserved0[1:0]` 提取 2-bit bbuf_idx。

```cpp
// idescriptor.h — operand_block_t
struct operand_block_t {
  uint32_t d_taddr;
  uint32_t a_taddr;
  uint32_t b_sdesc_lo;
  uint32_t b_sdesc_hi;
  uint16_t lanes_off;
  uint16_t reserved0;        // [1:0] = bbuf_idx (0-3)
  uint32_t fmt_cd;
  uint32_t reserved1[2];
};
```

## 3. Buffer 状态机

### 3.1 生命周期

```
INVALID ──FILL─────────────→ VALID (owner=warp X, compute_inflight=true)
INVALID ──USE──────────────→ stall (owner 不匹配或无数据)
INVALID ──LASTUSE──────────→ stall
INVALID ──DISCARD──────────→ 正常通过（AMem/BMem 路径）

VALID   ──FILL (owner 计算未完成) → stall
VALID   ──FILL (owner 计算已完成) → VALID (owner=新 warp, 直接覆盖)
VALID   ──USE (owner 匹配)────→ VALID (compute_inflight=true)
VALID   ──USE (owner 不匹配)──→ stall
VALID   ──LASTUSE (owner 匹配) → 计算完成后 → INVALID
VALID   ──LASTUSE (owner 不匹配)→ stall
VALID   ──DISCARD (owner 匹配, 计算未完成) → stall
VALID   ──DISCARD (owner 匹配, 计算已完成) → 计算完成后 → INVALID
VALID   ──DISCARD (owner 不匹配) → 正常通过（AMem/BMem 路径），不影响 buffer
```

### 3.2 CollectorState

```cpp
struct CollectorState {
    bool     valid = false;
    uint32_t owner_wid = 0;
    bool     compute_inflight = false;  // owner 的 MMA 计算是否还在 Stage3 中
};
```

## 4. 文件结构

### 4.1 新建文件

```
sim/simx/tensor/open_tensorcore/tensor_compute/collector_buffer.h
sim/simx/tensor/open_tensorcore/tensor_compute/collector_buffer.cpp
```

### 4.2 三个独立类

**ABuf** — A 矩阵 collector buffer，与 AMem 同维度（4 lines, 8×8 FP9, 8 banks）。

```cpp
class ABuf {
    static constexpr uint32_t kDepth = 4;
    static constexpr uint32_t kBankCount = 8;
    using row_t = std::array<std::array<uint16_t, 8>, 8>;
    std::array<row_t, kDepth> lines_;
    std::array<bool, kDepth>   line_valid_;
    CollectorState state_;

public:
    void clear();
    bool write_line(uint32_t line_idx, const uint16_t data[8][8]);
    void read_primitive(uint32_t line_idx, uint16_t out[8][8]) const;
    bool all_lines_valid() const;

    // Collector 生命周期
    void fill(uint32_t wid);
    void mark_compute_started();
    void mark_compute_done();     // 计算 retire 后调用
    void invalidate();            // DISCARD/LASTUSE retire → INVALID
    bool is_ready_for(uint32_t wid, uint8_t collector_buffer) const;
    uint32_t owner() const;
    bool valid() const;
};
```

**MBuf** — 稀疏 metadata collector buffer，与 MetaMem 同维度（1 packet, 64B）。

```cpp
class MBuf {
    std::array<uint8_t, 64> data_;
    CollectorState state_;

public:
    void clear();
    void write(const std::array<uint8_t, 64>& data);
    void read(uint8_t out[16], uint32_t step_m, uint32_t step_k) const;

    // Collector 生命周期（同 ABuf）
    void fill(uint32_t wid);
    void mark_compute_started();
    void mark_compute_done();
    void invalidate();
    bool is_ready_for(uint32_t wid, uint8_t collector_buffer) const;
    bool valid() const;
};
```

**BBuf** — B 矩阵 collector buffer，与 BMem 同维度（4 lines, 8×8 FP9, 8 banks）。

```cpp
class BBuf {
    // 存储与 ABuf 完全一致
    std::array<row_t, kDepth> lines_;
    std::array<bool, kDepth>   line_valid_;
    CollectorState state_;
    uint8_t index_;  // 0-3

public:
    // 接口与 ABuf 一致
};
```

### 4.3 修改文件

| 文件 | 改动 |
|------|------|
| `otc_types.h` | ConvertedTile: 新增 `collector_buffer`, `bbuf_idx` 字段 |
| `idescriptor.h` | `operand_block_t.reserved0` → bbuf_idx |
| `types.h` | `collector_a_fill` → `collector_buffer` 重命名 |
| `stage1_tcdecode.cpp` | 字段改名为 `collector_buffer`；ws=1 时从 `op_block.reserved0` 提取 bbuf_idx |
| `opentensorcore.h/.cpp` | 新增 `collector_ready()` 方法；Stage1 通过 OTC 查询 buffer 状态 |
| `stage3_compute.h/.cpp` | 持有 `ABuf abuf_`, `MBuf mbuf_`, `BBuf bbuf_[4]`；`fill_abc` 分流；`advance_compute` 选源 |
| `decode.cpp` | 注释更新 |

## 5. Pipeline 集成

### 5.1 Stage1 — Stall 逻辑

TcDecodeStage MMA 解码后、发 LsuReq 前，通过 `OpenTensorCore::collector_ready()` 检查就绪状态。不就绪 → 不 pop Input，不 issue LsuReq，下 tick 重试。

```cpp
// opentensorcore.h
bool collector_ready(uint32_t wid, uint8_t collector_buffer,
                     uint8_t ws, int8_t bbuf_idx) const;

// opentensorcore.cpp — 内部转发到对应的 ABuf 或 BBuf
bool OpenTensorCore::collector_ready(...) {
    if (ws == 0) {
        return abuf_.is_ready_for(wid, collector_buffer);
    } else {
        return bbuf_[bbuf_idx].is_ready_for(wid, collector_buffer);
    }
}
```

### 5.2 Stage2a — 跳过 TMEM/LMEM 读

USE/LASTUSE 时 abuf/bbuf 数据已在 Stage3 本地，Stage2a 跳过对应读取：

**ws=0 (控制 abuf)**：

| cb 值 | FETCH_A | FETCH_META | FETCH_C | FETCH_B | ConvertedTile A 段 |
|---|---|---|---|---|---|
| 00 FILL | 正常 | 正常 | 正常 | 正常 | 完整 A + collector_buffer |
| 01 USE | **跳过** | **跳过** | 正常 | 正常 | 空 + collector_buffer |
| 10 LASTUSE | **跳过** | **跳过** | 正常 | 正常 | 空 + collector_buffer |
| 11 DISCARD | 正常 | 正常 | 正常 | 正常 | 完整 A + collector_buffer |

**ws=1 (控制 bbuf)**：

| cb 值 | FETCH_A | FETCH_B | FETCH_C | ConvertedTile B 段 |
|---|---|---|---|---|
| 00 FILL | 正常 | 正常 | 正常 | 完整 B + collector_buffer + bbuf_idx |
| 01 USE | 正常 | **跳过** | 正常 | 空 + collector_buffer + bbuf_idx |
| 10 LASTUSE | 正常 | **跳过** | 正常 | 空 + collector_buffer + bbuf_idx |
| 11 DISCARD | 正常 | 正常 | 正常 | 完整 B + collector_buffer + bbuf_idx |

Note: ws 由 `bbuf_idx >= 0` 推导，无需在 ConvertedTile 中独立传递。

### 5.3 Stage3 — 读写分流

**fill_abc**：
- FILL abuf: A → abuf (不写 AMem), B → BMem, C → CMem
- USE/LASTUSE abuf: A 不写任何 SRAM（abuf 已有数据）
- FILL bbuf: B → bbuf[idx] (不写 BMem), A → AMem
- DISCARD: 走 AMem/BMem 正常路径

**advance_compute**：
- 读 A：`abuf_.valid() ? abuf_.read_primitive(...) : amem_.read_primitive(...)`
- 读 B：`bbuf_[idx].valid() ? bbuf_[idx].read_primitive(...) : bmem_[...].read_primitive(...)`

### 5.4 Stage3 — compute 完成后处理

LASTUSE/DISCARD retire 后调用 buffer 的 invalidate/mark_compute_done：

```cpp
// finish_job 中:
if (tile->collector_buffer == 0x2 /*LASTUSE*/ ||
    tile->collector_buffer == 0x3 /*DISCARD*/) {
    if (tile->bbuf_idx < 0) abuf_.invalidate();
    else bbuf_[tile->bbuf_idx].invalidate();
}
```

## 6. 性能收益

| 场景 | 无 collector | 有 collector |
|------|------------|------------|
| A 矩阵复用 3 次 | 3×(FETCH_A + META) = ~12 tick TMEM 读 | 1×FETCH_A + 2×跳过 = ~4 tick |
| B 矩阵复用 3 次 | 3×FETCH_B = ~12 tick LMEM 读 | 1×FETCH_B + 2×跳过 = ~4 tick |
| Sparse 多 chunk | 每 chunk 全量 A+META 读 | 首 chunk FILL，后续 USE（跳过 TMEM） |

collector buffer 的核心价值：将"重复 TMEM/LMEM 读"变成"首次 FILL + 后续 USE 零延迟"。

## 7. 边界条件

- **warp 切换**：每个 buffer 只能被一个 owner warp 持有。不匹配的 USE/LASTUSE → Stage1 stall
- **覆盖写入**：FILL 可直接覆盖已有 buffer——前提是 owner 计算已完成（compute_inflight=false）
- **compute_inflight**：从 FILL/USE/LASTUSE 的 fill_abc 开始标记 true，到 finish_job(LASTUSE/DISCARD) 或下次 FILL 覆盖后标记 false
- **DISCARD while compute_inflight**：必须 stall，等 compute 完成才能 reset buffer
- **DMem**：不受 collector buffer 影响，始终正常累加/写回
