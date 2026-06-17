# Vortex CModel —— TensorCore / TMEM / TMA / mbarrier / tcgen05 ISA 代码阅读指南

本文档面向**初次接触本 CModel 的阅读者**，目的是让你在不需要逐文件通读的前提下：

1. 知道每个新增模块在仓库中的物理位置；
2. 知道每个文件的"主功能"，避免把时间花在 legacy / 测试 / 调试代码上；
3. 拿到一份"按模块切片"的推荐阅读顺序；
4. 在读代码出现疑问时，能直接跳到关键函数/数据结构所在的行。

> 工程约定：所有新增模块仅在编译开关 `EXT_TCU_ENABLE` 打开时才被编入 `simx` 二进制；
> 实际进入构建的源文件以 `sim/simx/Makefile` 中 `EXT_TCU_ENABLE` 分支为准（见本文档"附录 A：构建产物"）。
> 凡未列入 Makefile 的同名源文件均视为**历史遗留（legacy）**，阅读时可跳过。

---

## 0. 总览：模块与目录的对应关系

```
vortex/
├── sim/common/
│   └── tensor_cfg.h                       # 张量扩展的全局配置（数据格式 id / WMMA 分块模板）
├── sim/simx/
│   ├── common/
│   │   ├── instr.h                        # 自定义 RISC-V opcode 枚举 EXT2 / EXT3 / EXT4
│   │   └── types.h                        # TcuType 枚举 + IntrTcuArgs（指令译码后字段）
│   ├── decode.cpp                         # EXT2/EXT3/EXT4 三个 opcode 的 decode 入口
│   ├── execute.cpp                        # 每种 TcuType 在 execute stage 的派发
│   ├── core.h / core.cpp                  # Core 侧的 mbarrier、TMA、TMEM、tcgen05.ld/st 状态机
│   └── tensor/                            # ★ 本次新增子系统的主目录 ★
│       ├── idescriptor.h                  # PTX 对齐的 5 种描述符结构体
│       ├── tensor_mem_port_types.h        # TensorUnit / TMA / TmemSystem 之间的端口消息
│       ├── tma.{h,cpp}                    # TMA：DRAM↔LMEM / LMEM↔TMEM 搬运
│       ├── tmem.{h,cpp}                   # TMEM：分配表 + 物理 bank/swizzle/仲裁
│       ├── tmem_system.{h,cpp}            # TMEM SimObject 顶层：port 仲裁 + tcgen05.shift 引擎
│       ├── tmem_math_packet.{h,cpp}       # TMEM packet 在数学 row-major 视图与物理 packet 视图之间转换
│       ├── tmem_utils.{h,cpp}             # TMEM 内部纯函数工具（bank/swizzle 计算、端口预算）
│       └── open_tensorcore/               # TensorCore 主体（前端控制 + 计算阵列）
│           ├── tensor_unit.{h,cpp}        # ★ TensorUnit 顶层 SimObject（Pimpl）★
│           ├── tensor_compute/            # 算术核心（fp 算子 + 8×8 阵列）
│           │   ├── fp_types.h             # 基础浮点工具（FP4/FP8/FP9/FP16/FP22/FP32 pack/unpack）
│           │   ├── fmul_s1/s2/s3.h        # 乘法器三级（s1：解析 / s2：尾数乘 / s3：规格化舍入）
│           │   ├── fadd_s1/s2.h           # 加法器两级
│           │   ├── tc_mul_pipe.h          # 3 级乘法流水封装
│           │   ├── tc_add_pipe.h          # 2 级加法流水封装
│           │   ├── tc_add4_pipe_fp22.h    # 4 输入 FP22 加法树（4 级）
│           │   ├── tc_mul_add.h           # 单元素 (i,j) 点积累加单元（含 accum_fifo）
│           │   ├── tensor_core_top.h      # ★ 8×8 阵列顶层 ★
│           │   ├── amem.h / bmem.h        # A / B 操作数 SRAM
│           │   ├── cmem.h / dmem.h        # C 输入 / D 输出 SRAM
│           │   ├── meta_mem.h             # 稀疏元数据存储
│           │   ├── sparse_select.h        # 2:4 / 1:4 稀疏路由
│           │   ├── fp22_to_fp16.h         # FP22→FP16 单独工具头
│           │   └── ARCHITECTURE.md        # 本子目录的官方说明书（推荐先看）
│           ├── tensor_control/
│           │   ├── tc_decode.{h,cpp}      # idesc + operand_block_t 译码 → TcDecodedMmaCmd
│           │   ├── config_register.h      # 全局 Config + TensorCoreMeta（贯穿流水的元数据）
│           │   ├── tcissue_unit.*         # ⚠ legacy（不在构建中）
│           │   ├── tcwmma_retire_unit.*   # ⚠ legacy
│           │   └── mem_controller.*       # ⚠ legacy
│           └── tensor_helper/
│               ├── tensor_debug_utils.{h,cpp}  # op_string 等调试工具（进入构建）
│               ├── tensor_core_test_utils.h    # standalone 测试 helper（不进入主时序）
│               ├── tensor_mem_test_utils.h     # 本地 SRAM 测试 helper
│               └── legacy/, test/              # ⚠ legacy / 单测
├── kernel/include/
│   └── vx_tensor.h                        # kernel 侧 tcgen05 ISA 的 inline asm 包装
└── hw/
    ├── VX_config.h / VX_types.h           # 硬件常量；与 cmodel 之间共享的宏
    └── rtl/VX_gpu_pkg.sv                  # 与 C++ Opcode::EXT* 对应的 RTL 编码
```

---

## 1. TensorCore 模块（8×8 矩阵乘累加 + 近核前端）

TensorCore 包含两层：
- **前端控制器（TensorUnit）**：接收 `tcgen05.mma` 宏指令、做 TMEM↔本地 SRAM 的数据搬运、把 16×16 宏 WMMA 拆成 4×2=8 条 8×8 原语下发；
- **计算阵列（TensorCoreTop）**：64 个 `tc_mul_add` 单元构成的 8×8 FP9·FP9→FP22 点积阵列。

### 1.1 推荐阅读顺序

| 顺序 | 文件 | 看什么 |
|---|---|---|
| 1 | `sim/simx/tensor/open_tensorcore/tensor_compute/ARCHITECTURE.md` | 整个 `tensor_compute/` 子目录的官方说明 |
| 2 | `tensor_compute/fp_types.h` | FP9 / FP22 等内部格式 + `PrecisionType` 枚举 |
| 3 | `tensor_control/config_register.h` | `TensorCoreMeta`：贯穿流水的元数据结构 |
| 4 | `tensor_compute/tensor_core_top.h` | 8×8 阵列顶层（`push_uop` / `tick` / `pop_retired`） |
| 5 | `tensor_compute/tc_mul_add.h` | 单元素流水线（含 `accum_fifo` 的弹性缓冲规则） |
| 6 | `tensor_compute/{amem,bmem,cmem,dmem,meta_mem}.h` | 本地 SRAM 模型（一次性看完，结构对称） |
| 7 | `tensor_compute/sparse_select.h` | 2:4 / 1:4 稀疏路由 |
| 8 | `tensor/open_tensorcore/tensor_unit.h` | TensorUnit 对外接口（`dispatch_tcu_mma` 等） |
| 9 | `tensor/open_tensorcore/tensor_unit.cpp` | Pimpl `Impl`：`MmaOp` 状态机（FillA→FillMeta→FillB→FillC→Compute→StoreD→Complete）|
| 10 | `tensor_control/tc_decode.{h,cpp}` | idesc + operand_block_t 译码 |

### 1.2 关键文件功能速查

- **`tensor_unit.h`（341 行）** —— 公开 SimObject 接口：
  - `ExeTraceData`：附在 instr_trace 上的执行追踪（`rd_write` / `retry`）。
  - `PerfStats`：~40 项性能/阻塞计数器（注释里有逐字段含义）。
  - `IssueBlockReason`：发射阻塞原因枚举。
  - 端口：`TensorMemReqOut` / `TensorMemRspIn`（对接 TmemSystem）、`TensorAsyncOpCompletionOut`（通知 Core）。
  - `dispatch_tcu_mma(wid, rs1, rs2, qualifier, trace_data)`：execute stage 调用的单一入口。

- **`tensor_unit.cpp`（911 行）** —— Pimpl 实现：
  - `Impl::MmaOp`（102 行）：一条宏 MMA 的完整状态。
  - `Impl::tick()`（154 行）：每周期推进队首 MmaOp。
  - `advance_fill_a / fill_meta / fill_b / fill_c`：4 路 fill 路径，前 3 路走 TmemSystem，FillB 走 LMEM 直读 b_sdesc。
  - `advance_compute`（721 行）：每周期最多发射 1 条 8×8 原语；同时从 `TensorCoreTop` 取退休结果写入 DMem。
  - `advance_store_d`（804 行）：DMem→TMEM 写回。
  - `complete_front_mma`（835 行）：向 Core 发出 `TensorAsyncOpCompletion`。

- **`tensor_core_top.h`（265 行）** —— 8×8 阵列：
  - `push_uop(a,b,c,meta)`：把一条原语装入 staging。
  - `tick(out_ready)`：四阶段处理（广播→各 PE tick→清 staging→完成检测）。
  - 支持非输出驻留 / 输出驻留两种模式（`circulating` 控制 final_add 的 C 操作数来源）。

- **`tc_mul_add.h`** —— 单元素流水（共 ~12 拍）：
  - `mul_pipe×8`（3 拍）→ `add4_pipe_fp22×2`（4 拍）→ `merge_add`（2 拍）→ `final_add`（2 拍）→ 输出寄存器 1 拍。
  - `accum_fifo`：2-entry elastic buffer，输出驻留模式时替代 c_in 透传链供给 C。

- **`tc_decode.h/cpp`** —— Phase-2 只译 `TCU_MMA`：
  - `decode_tcu_mma(core, wid, rs1_value, rs2_value, qualifier, out)`：
    1. `rs1_value` 强转为 `idescriptor_t`，解出 kind/shape/sparsity 等位段；
    2. `core->lmem_read` 读 32 B `operand_block_t`（a_taddr / b_sdesc / d_taddr / fmt_cd / lanes_off）；
    3. `qualifier` 解出 enable_input_d / ws / sp / cta_group / collector / multicast 6 个修饰位；
    4. 输出 `TcDecodedMmaCmd`。

---

## 2. TMEM 模块（Tensor Memory）

PTX `tcgen05` 的 scratchpad，是一块**按 lane × col_byte 寻址、内部 banked / swizzled 的 SRAM**。

### 2.1 推荐阅读顺序

| 顺序 | 文件 | 看什么 |
|---|---|---|
| 1 | `tensor/idescriptor.h` | 5 种描述符：`idescriptor_t`, `operand_block_t`, `cpabulk_transfer_args_t`, `tensor_map_t`, `mbarrier_state_t` |
| 2 | `tensor/tensor_mem_port_types.h` | TensorUnit↔TmemSystem 的端口消息（`TensorMemPortReq/Rsp`, `TensorAsyncOpCompletion`） |
| 3 | `tensor/tmem.h` | `Tmem` 类公开 API、容量常量（128 lane × 512 byte = 64 KB）、`TmemAllocation` |
| 4 | `tensor/tmem.cpp` | 分配/释放、字节读写、packet 仲裁、shift_down 实现 |
| 5 | `tensor/tmem_system.h` | `TmemSystem` 顶层 SimObject：5 个 SimPort + tcgen05.shift 引擎 |
| 6 | `tensor/tmem_system.cpp` | port FIFO 驱动 + shift_transaction 状态机 + visible state 双缓冲 |
| 7 | `tensor/tmem_utils.h/cpp` | 纯函数：bank 互素步长、swizzle 计算、端口预算 |
| 8 | `tensor/tmem_math_packet.h/cpp` | packet 在数学行优先视图 ↔ 物理 packet 视图之间的 pack/unpack |

### 2.2 关键设计点

- **逻辑布局**（`tmem.h` 第 89–105 行）：
  - 1 个 packet = 64 B；1 列 = 512 B = 8 个 packet；总共 128 列 → 64 KB。
  - `kDefaultPhysicalBanks = 16`；通过 `bank_swizzle_base_stride_` 与 `bank_swizzle_lane_stride_`（互素）做 banking。
  - 读/写端口预算：每周期 1 read packet + 1 write packet，每 bank 各 1 端口。

- **TADDR 编码**（PTX 对齐）：
  - `taddr[15:0]` = lane base，`taddr[31:16]` = col_byte。
  - `0xFFFFFFFF` 是 alloc 失败的 sentinel（`Tmem::kInvalidTaddr`）。
  - `find_allocation_by_lane(lane, &col_base)` 反向查询分配表。

- **可见状态双缓冲**：
  - `payload_ready` / `meta_ready` 与 `visible_payload_ready` / `visible_meta_ready` 一一对应；
  - 每个仿真 tick 在 `TmemSystem::publish_visible_state()` 中把"live"快照成"visible"，避免同一周期内的写在调度时被读到。
  - `Core::tmem_handle_load_block_reason` 在 `core.cpp:820` 附近**只读 visible 副本**，决定是否阻塞发射。

- **端口仲裁链路**：
  - 客户端：`TensorUnit` 通过 `TensorMemReqOut` 发起；`Tma` 通过 `TmemReqOut` 发起；
  - `TmemSystem::tick()` 调用 `drain_one_inbound_request` 把请求转 `Tmem::enqueue_port_request`；
  - 仲裁与授权在 `Tmem::arbitrate_requests` / `request_granted` / `consume_request_grant`；
  - `TmemSystem::complete_granted_port_requests` 把数据搬回响应端口。

- **`tcgen05.shift` 引擎**（`tmem_system.cpp:133–259`）：
  - 由 `TmemSystem::issue_shift` 创建 `ShiftTransaction`；
  - 每周期为单个 packet 同时发起一对 read+write 请求，最后调 `region_shift_down` 完成行偏移；
  - 完成后通过 `AsyncOpCompletionOut` 通知 Core。

---

## 3. TMA 模块（Tensor Memory Accelerator）

TMA 负责 **DRAM ↔ LMEM** 与 **LMEM ↔ TMEM** 两类批量搬运。

### 3.1 文件清单与阅读顺序

| 顺序 | 文件 | 看什么 |
|---|---|---|
| 1 | `tensor/idescriptor.h` 中的 `tensor_map_t` / `cpabulk_transfer_args_t` | 数据布局（128 B / 32 B） |
| 2 | `tensor/tma.h` | `Tma` SimObject、`TmaCpAsyncBulkResult`、`PendingLmemToTmemCopyOp` |
| 3 | `tensor/tma.cpp` | `cpabulk_tensor_load/store`（同步 DRAM↔LMEM）+ `issue_lmem_to_tmem_copy`（异步 LMEM→TMEM packet 状态机） |
| 4 | `sim/simx/core.cpp:1537–1620` 附近 | `Core::cpabulk_tensor_load/store`：把 TMA 结果挂上 `AsyncTensorOp` 并联到 mbarrier |

### 3.2 三种数据流

1. **DRAM → LMEM** —— `cp.async.bulk.tensor.<N>d` LD
   - 入口：`tcu` ISA `CPABULK_TENSOR_LD`（custom-1, funct3=0b110）；
   - 执行：`Tma::cpabulk_tensor_load(tensor_map_addr, args_lmem_ptr, complete_tx)`；
   - 实现：读 128 B `tensor_map_t`（DRAM）+ 32 B `cpabulk_transfer_args_t`（LMEM），用 64 B chunk 串行 `dcache_read` + `lmem_write`；
   - 当 `qualifier[5]=1` 时返回 `tx_bound_mbar/tx_bytes`，由 `Core::on_async_tensor_op_completed` 调用 `mbarrier_complete_tx` 兑现（PTX §7.6.4）。

2. **LMEM → DRAM** —— `cp.async.bulk.tensor.<N>d` ST
   - 对应函数：`Tma::cpabulk_tensor_store`（`lmem_read` + `dcache_write`）。

3. **LMEM → TMEM** —— `tcgen05.cp` 路径
   - 入口：`tcu` ISA `TMEM_CP`（custom-1, funct3=0b100）；
   - 执行：`Tma::issue_lmem_to_tmem_copy` 注册 `PendingLmemToTmemCopyOp`；
   - 周期推进：`Tma::advance_lmem_to_tmem_copy_ops()`，按 packet 依次 read-modify-write（packet 内 byte_offset 非对齐时需要先读再合并）；
   - 完成：拉 `payload_ready=true` + 推 `AsyncOpCompletionOut`，再由 Core 解锁等待的 warp。

### 3.3 端口拓扑（在 `core.cpp` 的 Core 构造函数里绑定）

```
TensorUnit.TensorMemReqOut ──▶ TmemSystem.TensorExecuteReqIn
TmemSystem.TensorExecuteRspOut ──▶ TensorUnit.TensorMemRspIn

Tma.TmemReqOut ──▶ TmemSystem.CoreTransferReqIn
TmemSystem.CoreTransferRspOut ──▶ Tma.TmemRspIn

TensorUnit.TensorAsyncOpCompletionOut ──▶ Core.tensor_async_op_completion_in_
Tma.AsyncOpCompletionOut               ──▶ Core.tma_async_op_completion_in_
TmemSystem.AsyncOpCompletionOut        ──▶ Core.tmem_system_async_op_completion_in_
```

> 上述绑定在 `sim/simx/core.cpp:93-101`。

---

## 4. mbarrier 异步执行

### 4.1 状态在哪？

`mbarrier` 对象**双重存在**：
- **Core 侧的镜像**：`Core::MBarrierEntry`（见 `core.h:499-509`），按 LMEM 地址索引：
  ```cpp
  std::unordered_map<uint64_t, MBarrierEntry> mbarriers_;
  ```
- **LMEM 中可被 kernel 直接读到的 8 B 物理字段**：`mbarrier_state_t`（`idescriptor.h:169`）。
  - 每次 Core 侧状态变化都通过 `mirror_mbarrier_to_lmem`（`core.cpp:862`）按 PTX §7.6.1 位布局回写。

### 4.2 完整 API（实现位置全部在 `sim/simx/core.cpp`）

| 函数 | 行号 | 对应 PTX |
|---|---|---|
| `mbarrier_init(addr, count)`            | 1376 | `mbarrier.init`             |
| `mbarrier_invalidate(addr)`             | 1392 | `mbarrier.invalidate`       |
| `mbarrier_arrive(addr, dec=1) → phase`  | 1412 | `mbarrier.arrive`           |
| `mbarrier_arrive_drop(addr)`            | 1431 | `mbarrier.arrive_drop`      |
| `mbarrier_expect_tx(addr, bytes)`       | 1448 | `mbarrier.expect_tx`        |
| `mbarrier_complete_tx(addr, bytes)`     | 1459 | `mbarrier.complete_tx`      |
| `mbarrier_wait(wid, addr, phase_token)` | 1472 | `mbarrier.wait.parity`      |
| `mbarrier_test_wait(addr, phase_token)` | 1503 | `mbarrier.test_wait`        |
| `mbarrier_try_wait(...)`                | 1515 | `mbarrier.try_wait`         |
| `try_complete_mbarrier(addr)`           | 874  | phase advance（内部辅助）   |

### 4.3 异步流水如何"汇合"到 mbarrier

`Core::on_async_tensor_op_completed`（`core.cpp:903`）是异步完成的统一钩子：

1. 若 `op.tx_bound_mbar != 0` —— 这是 `cp.async.bulk.tensor.mbarrier::complete_tx::bytes` 路径：
   立刻调 `mbarrier_complete_tx(op.tx_bound_mbar, op.tx_bytes)`（PTX §7.6.4）。
2. `resume_async_waiters(op.async_id)` —— 唤醒 `TMA_WAIT` / `TC_FENCE` 等显式等待者。
3. 若 `op.committed == true` —— 这是 `tcgen05.commit` 路径（PTX §9.7.16.5.7）：
   把 `op.barrier_id`（在 `tc_commit()` 时被重新解读为 LMEM mbar 地址）所指 barrier
   的 `pending_arrival_count` 减一。
4. 调 `try_complete_mbarrier(addr)`：若 `pending_arrival == 0 && pending_tx ≥ expected_tx`，
   翻转 phase 奇偶位、重装 `pending_arrival_count = expected_arrival_count`、清空 tx 计数，
   并通过 `emulator_.resume(wid)` 唤醒所有 `waiters_bitmap` 中的 warp。

> **要点**：`tcgen05.commit` 触发的是 **arrival**（次数计数），不是 **complete_tx**（字节计数）。
> 两者共同决定 phase 推进 —— 这是 PTX 的双计数器设计。

### 4.4 等待路径如何挂起 warp

- `mbarrier_wait`：把当前 warpgroup 全部 wid 写入 `mbarrier_wait_targets_[wid][addr] = phase_token`
  且置位 `barrier.waiters_bitmap`，然后返回 false → execute 层 `core_->suspend(wid, WarpStallReason::MBarrier)`。
- 当 `try_complete_mbarrier` 触发 phase advance，遍历 `waiters_bitmap` 对每个 wid 调 `emulator_.resume(wid)`。
- 非阻塞变体 `test_wait` 仅做 phase parity 比较即返回。

---

## 5. tcgen05 兼容的 RISC-V 扩展指令集

### 5.1 整体编码

| Opcode 名 | 数值 | 字段 | 用途 |
|---|---|---|---|
| `EXT2` / RISCV_CUSTOM1 | `0x2B` | funct3 + funct7 | TMEM 管理 + `cp.async.bulk.tensor` |
| `EXT3` / RISCV_CUSTOM2 | `0x5B` | funct3 + funct7 | tcgen05 sync + full mbarrier |
| `EXT4` / RISCV_CUSTOM3 | `0x7B` | funct3 + funct7 | tcgen05 compute（mma / ld / st / wait） |

- C++ 侧定义：`sim/simx/common/instr.h:48-51`（`enum class Opcode`）。
- 硬件侧定义：`hw/rtl/VX_gpu_pkg.sv:159-161`（`localparam INST_EXT2/3/4`）。
- 编译时宏：`kernel/include/vx_intrinsics.h:35-37` 的 `RISCV_CUSTOM1/2/3`。

### 5.2 指令清单（funct3 → TcuType）

| Opcode | funct3 | TcuType | 备注 |
|---|---|---|---|
| EXT2 | `0b001` | `TMEM_REL_PERMIT`    | `tcgen05.relinquish_alloc_permit` |
| EXT2 | `0b010` | `TMEM_ALLOC`         | rd=handle, rs1=col_span, rs2=reserved(必须传 `0xffffffff`) |
| EXT2 | `0b011` | `TMEM_DEALLOC`       | rs1=handle |
| EXT2 | `0b100` | `TMEM_CP`            | rs1=taddr, rs2=LMEM ptr→64b s_desc |
| EXT2 | `0b101` | `TMEM_SHIFT`         | rs1=handle, rs2=control |
| EXT2 | `0b110` | `CPABULK_TENSOR_LD`  | rs1=tensor_map DRAM ptr, rs2=LMEM ptr→cpabulk_transfer_args_t |
| EXT2 | `0b111` | `CPABULK_TENSOR_ST`  | 同上但反向 |
| EXT3 | `0b000` | `MBAR_FENCE`         | qualifier[0]=before(0)/after(1) |
| EXT3 | `0b001` | `MBAR_COMMIT`        | rs1=mbar_addr, rs2=cta_mask |
| EXT3 | `0b010` | `MBAR_INIT`          | qualifier[0]=invalidate |
| EXT3 | `0b011` | `MBAR_ARRIVE`        | qualifier[1]=drop, [3]=expect_tx_combo |
| EXT3 | `0b100` | `MBAR_EXPECT_TX`     | rs2=tx_bytes |
| EXT3 | `0b101` | `MBAR_COMPLETE_TX`   | rs2=tx_bytes |
| EXT3 | `0b110` | `MBAR_WAIT`          | rs2=phase_token，阻塞 |
| EXT3 | `0b111` | `MBAR_TEST_TRY_WAIT` | qualifier[0]=test(0)/try(1) |
| EXT4 | `0b000` | `TCU_MMA`            | rs1=idesc(32 bit), rs2=LMEM ptr→operand_block_t |
| EXT4 | `0b001` | `TCU_LD`             | rd=data, rs1=taddr |
| EXT4 | `0b010` | `TCU_ST`             | rs1=taddr, rs2=value |
| EXT4 | `0b011` | `TCU_WAIT_LD`        | 等待本 warp 在飞的 tcgen05.ld |
| EXT4 | `0b100` | `TCU_WAIT_ST`        | 等待 tcgen05.st |

> qualifier 各位段含义在 `sim/simx/common/types.h:776-836` 的 `IntrTcuArgs` 注释里。
> 标准 PTX `idesc` 32 位位段在 `tensor/idescriptor.h:39-67`。

### 5.3 三条调用链（从 kernel 到 cmodel）

#### a) kernel 端（用户写的 C++）

`kernel/include/vx_tensor.h` 用 `__asm__ volatile (".insn r ...")` 包了一套 inline intrinsics：

- `tmem_alloc / tmem_dealloc / tmem_cp / tmem_shift / cpabulk_tensor_ld / cpabulk_tensor_st`（EXT2）
- `mbar_fence_before/after / mbar_commit / mbarrier_init / mbarrier_arrive / mbarrier_arrive_expect_tx / mbarrier_expect_tx / mbarrier_complete_tx / mbarrier_wait / mbarrier_test_wait / mbarrier_try_wait`（EXT3）
- `tcu_mma / tcu_mma_no_accum / tcu_ld / tcu_st / tcu_wait_ld / tcu_wait_st`（EXT4）

还有几个 host/device 共用的"构造器"：`make_idescriptor`, `make_operand_block`, `make_cpabulk_args`（位于 `vx_tensor.h` 上半段）。

#### b) decode（`sim/simx/decode.cpp`）

EXT2/EXT3/EXT4 在 `decode.cpp:1111-1305` 集中处理。每个 case 都把 funct7 切片成 `IntrTcuArgs` 的字段，
然后 `make_tcu_instr(instr_pool_, uuid, TcuType::XXX, args)` 投递到 ibuffer。

#### c) execute（`sim/simx/execute.cpp:1458-1696`）

按 `TcuType` switch，每条指令做的事情用一行描述：

```
TMEM_ALLOC          rd = core_->tmem_alloc(col_span, reserved)
TMEM_REL_PERMIT     core_->tmem_rel_permit()
TMEM_DEALLOC        core_->tmem_dealloc(handle)
TMEM_CP             core_->tmem_cp(wid, taddr, s_desc_lmem_ptr, shape, decompress)
TMEM_SHIFT          rd = core_->tmem_shift(wid, handle, ctl)
CPABULK_TENSOR_LD   rd = core_->cpabulk_tensor_load(wid, tmap, args_ptr, complete_tx)
CPABULK_TENSOR_ST   rd = core_->cpabulk_tensor_store(wid, tmap, args_ptr)

MBAR_FENCE          core_->mbar_fence(wid, mode) [若 false → suspend]
MBAR_COMMIT         rd = core_->mbar_commit(wid, mbar_addr, cta_mask)
MBAR_INIT           core_->mbarrier_init(mbar_addr, count)
MBAR_ARRIVE         rd = core_->mbarrier_arrive(mbar_addr)
MBAR_EXPECT_TX      core_->mbarrier_expect_tx(mbar_addr, bytes)
MBAR_COMPLETE_TX    core_->mbarrier_complete_tx(mbar_addr, bytes)
MBAR_WAIT           core_->mbarrier_wait(wid, addr, token) [若 false → suspend]
MBAR_TEST_TRY_WAIT  rd = mbarrier_{test,try}_wait(...)

TCU_MMA             tensor_unit_->dispatch_tcu_mma(wid, rs1, rs2, qualifier, trace) [retry → 重发]
TCU_LD              core_->issue_tcgen05_ld_async(wid, trace_id, dst, tmask, taddrs)
TCU_ST              core_->issue_tcgen05_st_async(wid, trace_id, tmask, taddrs, values)
TCU_WAIT_LD         core_->tcgen05_wait_ld(wid)  [若 false → suspend]
TCU_WAIT_ST         core_->tcgen05_wait_st(wid)  [若 false → suspend]
```

`TCU_MMA` 是唯一会"分发到 `TensorUnit`"的指令，其余都直接打在 Core 上。
`TCU_LD/ST` 是**逐线程**的 TMEM 访问，由 Core 维护 `pending_tcgen05_ldst_ops_` 状态机（`core.h:483-494`、`core.cpp` 的 `advance_tcgen05_ldst_async_ops`）。

---

## 6. 跨模块串起来的一条端到端路径

为了让阅读者把"零散的文件"和"一条 PTX 风格的内核"对应起来，下面用 `cp.async.bulk → tcgen05.mma → tcgen05.st` 三个阶段示意端到端调用链。

### 阶段 1 — 把数据从 DRAM 异步拉到 LMEM，由 mbarrier 通知
```
kernel:  cpabulk_tensor_ld_complete_tx(tmap_ptr, args_lmem_ptr);   // vx_tensor.h
         mbarrier_wait(mbar_addr, phase=0);
         ─────────────────────────────────────────────────────────
decode:  EXT2/funct3=0b110 → TcuType::CPABULK_TENSOR_LD            // decode.cpp
         EXT3/funct3=0b110 → TcuType::MBAR_WAIT
execute: Core::cpabulk_tensor_load(...)                            // execute.cpp:1542
           └─ Tma::cpabulk_tensor_load(tensor_map_addr,
                                      args_lmem_ptr, complete_tx)  // tma.cpp
                ├─ dcache_read(tmap, 128 B)
                ├─ dcache_read(src, N×64 B) + lmem_write(dst, ...)
                └─ 返回 {tx_bound_mbar, tx_bytes}
         AsyncTensorOp{type=TmaLoad, tx_bound_mbar, tx_bytes}.completed=true
         Core::on_async_tensor_op_completed → mbarrier_complete_tx
           → try_complete_mbarrier → phase 翻转 → emulator_.resume(wid)
         Core::mbarrier_wait 返回 true（被 resume）
```

### 阶段 2 — 在 TMEM 中执行 16×16 MMA
```
kernel:  uint32_t d_taddr = tmem_alloc(col_span);
         uint32_t idesc   = make_idescriptor<...>(...);
         operand_block_t ob = make_operand_block<...>(a_taddr, b_sdesc, ...);
         tcu_mma(d_taddr, idesc, &ob);                              // EXT4/funct3=000
         ─────────────────────────────────────────────────────────
decode:  TcuType::TCU_MMA                                          // decode.cpp:1253
execute: tensor_unit_->dispatch_tcu_mma(wid, rs1, rs2, qual, td);  // execute.cpp:1622
TensorUnit (tensor_unit.cpp):
  Impl::enqueue_tcu_mma
    ├─ TcDecode::decode_tcu_mma(...)                               // 解 idesc + lmem_read operand_block_t
    ├─ Core::tmem_handle_load_block_reason(...)（A/D 槽位）        // 看 visible_payload_ready
    ├─ Core::wmma_async_issue(wid) → async_id
    └─ pending_mma_.push_back(MmaOp{...})
  Impl::tick → advance_mma →
    FillA: 向 TmemSystem 发 N 个 RegionRead packet 请求 → 进 AMem
    FillMeta（sparse 才有）: 再读 1 个 packet → MetaMem
    FillB: 走 Core::lmem_read 直接从 shared 拉 B → BMem
    FillC（enable_input_d 才有）: 读 d_taddr 一遍 → CMem
    Compute: 每周期最多 1 条原语下发到 TensorCoreTop；同步从其 retire 端取结果写 DMem
    StoreD: DMem→TMEM region write
    Complete: 向 Core 推 TensorAsyncOpCompletion(async_id)
Core::drain_tensor_execute_completion_notices → async_tensor_complete → on_async_tensor_op_completed
   → 若该 op 被 tcgen05.commit 过 → mbarrier_arrive
```

### 阶段 3 — warp 把 D 从 TMEM 取到寄存器
```
kernel:  uint32_t v = tcu_ld(taddr);                                // EXT4/funct3=001
         tcu_wait_ld();                                             // EXT4/funct3=011
         ─────────────────────────────────────────────────────────
execute: Core::issue_tcgen05_ld_async(wid, trace_id, dst, tmask, taddrs);
         Core::advance_tcgen05_ldst_async_ops 把每个线程的 (taddr.lane+t, col_byte..+3)
           映射到 packet_idx 并向 TmemSystem 发 RegionRead；
         所有线程完成后 tcgen05_ld_trace_ready(trace_id)=true，commit stage 写回 RF。
TCU_WAIT_LD: 若仍有未完成 → suspend(wid, WarpStallReason::AsyncTensor)，
             直到 try_resume_tcgen05_ldst_waiters 将其唤醒。
```

---

## 7. 阅读时常见的"坑"

1. **legacy 文件**：`tensor_control/tcissue_unit.*`、`tcwmma_retire_unit.*`、`mem_controller.*`
   以及 `tensor_helper/legacy/`、`tensor_helper/test/` 这些目录里的 `main.cpp` —— **均不在主构建里**，
   `include` 路径与名字也不再匹配主分支（你会看到 `#include "open_tensorcore/tensor_top/tensor_unit.h"`
   这种已经被废弃的层级）。先看 `Makefile` 的 EXT_TCU_ENABLE 分支确认源文件白名单。

2. **同名头文件多版本**：`config_register.h` 中只有 `Config` 与 `TensorCoreMeta` 仍被主路径使用；
   其它 legacy 头里的"状态机风格"接口（`tud::AMemState` 等）已经被 `tensor_unit.cpp` 内部的 `MmaOp` 整合替代。

3. **TMEM 与 LMEM 千万别弄混**：
   - LMEM = `LocalMem`（每核共享 scratchpad，kernel 普通 load/store 指令可访问，mbarrier 对象住在这里）。
   - TMEM = `Tmem`（PTX `tcgen05` 专用 scratchpad，只能通过 `tcgen05.ld/st/cp/mma/shift` 访问）。
   - `Core::lmem_read/write`（`core.h:209`）和 `Tmem::handle_region_read/write_bytes` 是两套独立 API。

4. **`tcgen05.commit` 的 barrier_id 不是普通整数**：
   在 `core.cpp` 的 `mbar_commit` 实现里，barrier_id 字段被**重解释为 LMEM mbar 地址**保存在
   `AsyncTensorOp::barrier_id`，完成时再当地址使用（`on_async_tensor_op_completed`）。

5. **可见状态与 live 状态的差异**：所有调度判断（`tmem_handle_load_block_reason` 等）都读 visible 副本；
   修改 ready 标志时会有 `update_visible_now=true` 参数控制是否立即同步，默认 false（等下一次
   `publish_visible_state` 才生效）。读源码时若发现"我刚刚设了 ready 为 true 怎么调度还看不到"，去查这个 flag。

6. **CModel 中 `try_wait` 没有真正的超时计数**：当前实现等同 `test_wait`（见 `core.cpp:1515-1523`）。
   若 kernel 期望非零 `timeout_bucket` 的行为，需要后续补 schedule wakeup。

---

## 附录 A：构建产物白名单

> 摘自 `sim/simx/Makefile` 的 `EXT_TCU_ENABLE` 分支。任何**不在此列表**的 `.cpp`/`.h` 都不会被
> 主仿真器编译。

```
sim/simx/tensor/tma.cpp
sim/simx/tensor/tmem_utils.cpp
sim/simx/tensor/tmem_math_packet.cpp
sim/simx/tensor/tmem.cpp
sim/simx/tensor/tmem_system.cpp
sim/simx/tensor/open_tensorcore/tensor_control/tc_decode.cpp
sim/simx/tensor/open_tensorcore/tensor_helper/tensor_debug_utils.cpp
sim/simx/tensor/open_tensorcore/tensor_unit.cpp
```

头文件的 include path：
```
sim/simx
sim/simx/common
sim/simx/tensor
sim/simx/tensor/open_tensorcore
sim/simx/tensor/open_tensorcore/tensor_compute
sim/simx/tensor/open_tensorcore/tensor_control
sim/simx/tensor/open_tensorcore/tensor_helper
sim/common
hw
```

---

## 附录 B：一张"看完即懂"速查图

```
                       ┌─────────────────────────────────────────────┐
        kernel  ──────▶│ vx_tensor.h 的 inline asm（EXT2/3/4）       │
                       └────────────────────┬────────────────────────┘
                                            │ 32-bit RISC-V .insn r
                                            ▼
                       ┌─────────────────────────────────────────────┐
                       │ decode.cpp: 切 funct3/funct7 → TcuType+args │
                       └────────────────────┬────────────────────────┘
                                            │ instr_trace_t
                                            ▼
                       ┌─────────────────────────────────────────────┐
                       │ execute.cpp: switch(TcuType)                │
                       └────────────────────┬────────────────────────┘
                       ┌───────────────┬────┴────┬─────────────────┐
                       ▼               ▼         ▼                 ▼
                 ┌──────────┐  ┌──────────────┐  ┌────────────┐  ┌────────────────┐
                 │ Core::   │  │ TensorUnit::│  │ Core::     │  │ Core::         │
                 │ mbarrier_│  │ dispatch_   │  │ cpabulk_   │  │ issue_tcgen05_ │
                 │ *()      │  │ tcu_mma()   │  │ tensor_*() │  │ ld/st_async()  │
                 └────┬─────┘  └──────┬──────┘  └─────┬──────┘  └──────┬─────────┘
                      │               │               │                │
                LMEM mirror     TmemSystem      Tma SimObject      Core 自维护
              mbarrier_state_t   (Tmem)        DRAM↔LMEM           pending_tcgen05_
                                  │           LMEM↔TMEM           ldst_ops_
                                  ▼               │                    │
                          packet 仲裁 + shift     │                    │
                                  │               │                    │
                                  ▼               ▼                    ▼
                          ┌────────────────────────────────────────────────────┐
                          │  AsyncOpCompletion → Core::on_async_tensor_op_     │
                          │  completed → mbarrier_complete_tx / mbarrier_      │
                          │  arrive → try_complete_mbarrier → resume(wid)      │
                          └────────────────────────────────────────────────────┘
```

阅读建议：先按本文档第 0 章在 IDE 里把目录结构定位好，然后按第 1–5 章一个模块一个模块地"边读 .h 边对照 .cpp"。
对疑问点遇到具体行号时，可直接用本文档列出的文件:行号在编辑器中跳转。

