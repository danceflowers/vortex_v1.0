# LMEM → TMEM 矩阵存储与取出分析

## 写入路径：LMEM → TMEM

以 `Tma::advance_lmem_to_tmem_copy_ops()` 为例（`tma.cpp:253`）：

```
LMEM (行主序存储的矩阵)
  │
  │  每次读 64B (kPacketBytes)
  │  lmem_read(packet.bytes, lmem_addr + cursor, 64)
  ▼
TmemPacket (64 字节数组, 顺序不变)
  │
  │  TensorMemPortReq → TmemReqOut
  ▼
Tmem::region_write_packet(col_base, col_span, packet_idx, packet)
  │
  │  offset = packet_idx * 64
  │  逐字节调用 write_region_byte(col_base, offset+i, packet.bytes[i])
  ▼
write_region_byte → write_logical_byte(col, line, value)
  │
  │  logical_col  = col_base + byte_offset / 512   (kColBytes=512)
  │  logical_line = byte_offset % 512
  ▼
logical_byte_to_physical(col, line, ...)
  │
  │  bank = line_chunk_bank(logical_line, chunk_idx, ...)  ← swizzle
  │  row  = logical_line / 2
  │  bb   = logical_col % 8
  ▼
banks_[bank][row][bb] = value   (16 个物理 bank, 每 bank 256 行 × 8B)
```

### TMA 的 LMEM → TMEM 状态机

在 `tma.cpp:253` 的 `advance_lmem_to_tmem_copy_ops()` 中，每次拷贝分两个阶段：

1. **Read 阶段**：从 LMEM 读 64B → `lmem_read(packet.bytes, lmem_addr + cursor, 64)` → 发 `RegionRead` 请求到 TMEM
2. **Write 阶段**：收到读响应后，把读到的数据发 `RegionWrite` 请求写入 TMEM

对于非对齐的包（起始偏移或剩余字节数不是 64 的整数倍），先走 Read-Modify-Write 流程。

---

## 读取路径：TMEM → 外部

```
Tmem::region_read_packet(col_base, col_span, packet_idx, &out)
  │
  │  offset = packet_idx * 64
  │  逐字节调用 read_region_byte(col_base, offset+i)
  ▼
read_region_byte → read_logical_byte(col, line)
  │
  │  同样的 logical_byte_to_physical 映射
  ▼
out.bytes[i] = banks_[bank][row][bb]
```

**读出的 64 字节顺序 = 写入时的 64 字节顺序**，完全透明。

---

## 逻辑布局：列主序（column-major）

`write_region_byte(col_base, byte_offset, value)` 的映射规则：

```
byte_offset 决定逻辑坐标:
  logical_col  = col_base + byte_offset / 512   ← 每 512 字节换一列(lane)
  logical_line = byte_offset % 512               ← 列内偏移，向下延伸
```

一个 col_span=16 的分配 (16 lanes × 512B = 8KB)：

```
  lane0      lane1      ...  lane15
  ──────     ──────          ──────
  line0      line0           line0       ← byte 0..15 分别落在 lane0..15 的 line0
  line1      line1           line1       ← byte 16..31 分别落在 lane0..15 的 line1
  ...        ...             ...
  line511    line511         line511     ← 共 512 行 × 16 列 = 8192 字节
```

对连续 64B 包（packet 0 = byte 0..63）：

```
byte 0 → lane0, line0
byte 1 → lane0, line1
byte 2 → lane0, line2
...
byte 63→ lane0, line63

所有 64 字节都在 lane0 中，沿 line 方向排布。
```

**关键点**：TMEM 内部是以 lane（逻辑列）为单位的，byte stream 按列主序填充——先把一个 lane 的 512 字节行填满，再移到下一个 lane。这与 LMEM 的行主序是不同的。

---

## 物理映射：Bank Swizzle

`logical_byte_to_physical` 的核心 swizzle 公式（`tmem_utils.cpp:155`）：

```
chunk_idx = logical_col / 8                    ← 每 8 字节宽为一组
bank      = line_chunk_bank(logical_line, chunk_idx, ...)
          = ((logical_line/2) * stride +         ← row 贡献
             (logical_line%2) * 8 +              ← slot 贡献
             chunk_idx) % bank_count             ← chunk 贡献
row       = logical_line / 2                    ← 两行逻辑 line 合并为 1 物理行
bank_byte = logical_col % 8                     ← 行内字节偏移
```

默认参数：16 banks, stride=5 (与 16 互质), 每 bank 8B 宽。

以 packet 0 前几字节为例：

| byte_offset | logical (col,line) | physical (bank,row,bbyte) |
|---|---|---|
| 0 | (0, 0) | (0, 0, 0) |
| 1 | (0, 1) | (8, 0, 0) |
| 2 | (0, 2) | (5, 1, 0) |
| 3 | (0, 3) | (13, 1, 0) |
| 4 | (0, 4) | (10, 2, 0) |

相邻字节被分散到不同 bank，同一 cycle 内多个访问才不会冲突。Bank stride 选取与 bank 数量互质的整数，保证所有 bank 均匀参与。

---

## TMEM 物理结构参数

| 参数 | 值 | 说明 |
|---|---|---|
| `kPayloadCols` | 128 lanes | 逻辑 lane (列) 数 |
| `kLogicalLines` / `kColBytes` | 512 | 每 lane 字节行数 |
| 总容量 | 128 × 512 = **64 KB** | |
| `kPacketBytes` | 64 B | 包大小 |
| `kPhysicalBankBytes` | 8 B | 每 bank 行宽 |
| `kPacketLanes` | 8 | 64/8，每包的 lane slice 数 |
| `kDefaultPhysicalBanks` | 16 | 默认物理 bank 数 |
| `kPhysicalRows` | 256 | 512/2，物理行数（2 逻辑行=1 物理行） |

Bank 数量可通过环境变量 `VORTEX_SIMX_TMEM_BANKS` 覆盖。

---

## 总结

| 维度 | 当前行为 |
|---|---|
| **LMEM→TMEM 顺序** | 纯线性拷贝，每 64B 一个包，包内字节顺序 = LMEM 原文顺序 |
| **TMEM 逻辑布局** | **列主序**：byte_offset 先填满 lane0 的 512 行，再填 lane1 |
| **TMEM 物理映射** | **Bank swizzle**：相邻 line → 不同 bank，避免端口冲突 |
| **取出顺序** | 透明还原：读出的包内字节顺序 = 写入时顺序 |
| **有无矩阵感知** | **无**。当前走 `LinearPacketStream` 语义，只是把矩阵当二进制流存 |

`TmemMathPacketLayout`（ALineNative/BLineNative/CSubtileNative/MathRowMajor）在 `tmem_math_packet.h/.cpp` 中已完整实现，但目前未被 mem_controller 或 tensor_unit 调用。它们将在未来用于 TMEM ↔ AMem/BMem 的矩阵感知 swizzle/reorder，使包顺序匹配 TensorCore 的消费图样以减少读取 stall。
