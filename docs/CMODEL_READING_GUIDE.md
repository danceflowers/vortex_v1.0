# Vortex CModel —— tensor 文件夹源码速查

> 编译开关：`EXT_TCU_ENABLE`。实际进入构建的源码以 `sim/simx/Makefile` 的 `EXT_TCU_ENABLE` 分支为准；
> 标"⚠ legacy"的文件不在主构建里，可以跳过。

涵盖范围：`sim/simx/tensor/` 下**全部 .h / .cpp 文件**，按主题分为五组：

1. TensorCore（前端 + 计算阵列）
2. TMEM
3. TMA
4. mbarrier 异步执行（位于 `core.h/cpp`，与 tensor 文件夹联动）
5. tcgen05 兼容的 RISC-V 指令（位于 `decode.cpp` / `execute.cpp` / `vx_tensor.h`，是 tensor 文件夹的"入口"）

末尾附"数据路径"与"控制路径"两节。

---

## 1. TensorCore 模块

> 一条 `tcgen05.mma` 在 cmodel 中的完整旅程：tc_decode 译码 → MmaOp 状态机搬数据 → TensorCoreTop 8×8 阵列计算 → 写回 TMEM。

### 1.1 顶层 SimObject（`tensor/open_tensorcore/`）

| 文件 | 主要功能 | 由谁调用 |
|---|---|---|
| `tensor_unit.h` | **TensorUnit SimObject 对外接口**。声明 3 个 SimPort（`TensorMemReqOut/RspIn` 对接 TmemSystem；`TensorAsyncOpCompletionOut` 对接 Core）、`ExeTraceData`（trace 上的 rd_write/retry 标志）、`PerfStats`（40+ 项计数器，注释里逐字段解释）、`IssueBlockReason` 枚举（11 种发射阻塞原因）、`dispatch_tcu_mma()` 唯一入口。Pimpl 模式，实现细节藏在 `Impl` 类。 | `Core` 构造时实例化 `tensor_unit_`；`execute.cpp::TCU_MMA` 调用 `dispatch_tcu_mma`；性能报告读 `perf_stats()` |
| `tensor_unit.cpp` | **TensorUnit Impl**。核心是 `MmaOp` 结构体（7 阶段：FillA→FillMeta→FillB→FillC→Compute→StoreD→Complete）+ 一个 deque `pending_mma_`，`tick()` 每周期推进队首。包含 7 个 `advance_*` 函数：FillA/Meta/C 走 TmemSystem 端口 packet read；FillB 走 `Core::lmem_read` 直读 b_sdesc；Compute 调 `TensorCoreTop::push_uop` 并从 retire 端写 DMem；StoreD 走端口 packet write；Complete 推 `TensorAsyncOpCompletion`。还有 `resolve_taddr`（TADDR→(col_base,byte_offset)）、`validate_mma`（不支持配置直接 abort）、`scheduler_score` 等。 | `TensorUnit` 公开方法 forward 到 Impl |

### 1.2 计算阵列（`tensor/open_tensorcore/tensor_compute/`，全部 header-only）

按 "数据类型 → 基础算子 → 流水封装 → 单元素 → 阵列" 五层组织。

#### a) 数据类型与基础工具

| 文件 | 主要功能 | 由谁调用 |
|---|---|---|
| `fp_types.h` | 张量扩展的浮点公共头。定义 `RoundingMode`（RNE/RTZ/RDN/RUP/RMM）、`PrecisionType` 枚举（FP4_E2M1/FP8_E4M3/FP8_E5M2/FP9/FP16/FP32）、各精度 pack/unpack（`convert_to_fp9`、`convert_c_to_fp22`、`fp22_to_fp32` …）、`clz` 等位级 helper。 | 所有 fmul/fadd 算子、tc_*_pipe、tensor_core_top、cmem/dmem、tc_decode |
| `fp22_to_fp16.h` | 独立的 FP22→FP16 转换函数（含 NaN/Inf/subnormal/round-to-nearest 处理）。 | `cmem.h` 读 fp16 C 输入时；`dmem.h` dump fp16 结果时 |

#### b) 浮点乘法器三级流水

| 文件 | 主要功能 | 由谁调用 |
|---|---|---|
| `fmul_s1.h` | 乘法器第 1 级。输入 FP9，做特殊值检测（NaN/Inf/zero）、计算 shift_amt 与 exp_shifted、识别 subnormal、决定 prod_sign，把 FP9 字段域扩展到内部 8/14 算术域。 | `tc_mul_pipe::tick` (r1 stage) |
| `fmul_s2.h` | 乘法器第 2 级。真正做尾数乘法，并把 s1 中间结果透传。 | `tc_mul_pipe::tick` (r2 stage) |
| `fmul_s3.h` | 乘法器第 3 级。规格化、舍入、特殊值封装，输出 FP22 乘积。 | `tc_mul_pipe::tick` (r3 stage) |

#### c) 浮点加法器两级流水 + 4 输入加法树

| 文件 | 主要功能 | 由谁调用 |
|---|---|---|
| `fadd_s1.h` | 加法器第 1 级。near/far path 选择、指数对齐、小数对齐、effective add/sub 判断。 | `tc_add_pipe`、`tc_add4_pipe_fp22` 内部 r1 |
| `fadd_s2.h` | 加法器第 2 级。规格化、舍入、特殊值打包。 | `tc_add_pipe`、`tc_add4_pipe_fp22` 内部 r2 |

#### d) 流水线封装（把基础算子包成统一 valid/ready 接口）

| 文件 | 主要功能 | 由谁调用 |
|---|---|---|
| `tc_mul_pipe.h` | 3 级乘法流水封装：把 `fmul_s1/s2/s3` 串成 r1/r2/r3 寄存器边界 + valid/ready 反压；同时透传 c_in 给后段 final_add（非输出驻留模式用）。 | `tc_mul_add.h` 里的 `mul_array[8]` |
| `tc_add_pipe.h` | 2 级 FP22 加法流水封装：`fadd_s1 → fadd_s2`。 | `tc_mul_add` 中的 `merge_add` 与 `final_add` |
| `tc_add4_pipe_fp22.h` | 4 输入 FP22 加法树（4 级）。把 4 个乘积归约为 1 个部分和；同时透传 passthrough 链路。 | `tc_mul_add` 中的 `add4_level[2]`，把 8 个乘积分两组各归约一次 |

#### e) 单元素 PE + 8×8 阵列

| 文件 | 主要功能 | 由谁调用 |
|---|---|---|
| `tc_mul_add.h` | **单个输出元素 D[i][j] 的完整流水线**。组件：8×mul_pipe（k=0..7 并行，3 拍）+ 2×add4_pipe_fp22（4 拍）+ 1×merge_add（2 拍）+ 1×final_add（2 拍）+ 输出寄存器（1 拍）。包含 **2-entry elastic `accum_fifo`**：输出驻留模式下替代 c_in 透传，由 CMem 预取的 C 填入，final_add 结果反馈回 FIFO tail。提供 `load_fifo` / `pop_fifo` / `tick`。 | `tensor_core_top.h` 实例化 64 个（M×N=8×8） |
| `tensor_core_top.h` | **8×8 阵列顶层**。持有 64 个 `tc_mul_add`、staging 缓冲区（`a_in/b_in/c_in/staged_meta`）、`retired` 输出寄存器、`circulating`（输出驻留开关）、`resident_tail_to_dmem_valid/async_id`（tail 写 DMem 用）。API：`push_uop(a,b,c,meta)` 装载一条原语；`tick(out_ready)` 四阶段处理（广播→PE tick→清 staging→完成检测）；`pop_retired(out)` 取出退休结果；`load_fifo_subtile` / `pop_fifo_element`（输出驻留模式的 C 预取与排空）。 | `tensor_unit.cpp::advance_compute` 唯一调用方 |

#### f) 操作数本地 SRAM（A/B/C/D + 稀疏 meta）

| 文件 | 主要功能 | 由谁调用 |
|---|---|---|
| `amem.h` | A 操作数 SRAM。深度 = 4 行（line 0/1=K-phase 0 的两个 M-block；line 2/3=K-phase 1）；每行 64 elem×fp9。提供 `write_fill_line(fmt, line, packets)`（fp16→fp9 / fp8→fp9 转换 + 写入）、`read_primitive(line_idx, out, transpose)`（输出 8×8 fp9）、`primitive_bank_mask`、`packet_count(fmt)`/`packets_per_fill_line(fmt)` 等容量计算。 | `tensor_unit.cpp::write_amem_lines`（FillA 末尾批量写）；`tensor_unit.cpp::issue_current_primitive`（Compute 阶段读） |
| `bmem.h` | B 操作数 SRAM。结构与 AMem 对称：4 行、8 bank。额外支持 sparse_mode 影响的 `packet_count(fmt, sparse_mode)` / `packets_per_fill_line(fmt, sparse_mode)`（稀疏时 B 物理量不变，密度按 A 的稀疏度等效折算）。 | `tensor_unit.cpp::advance_fill_b` 写入；`issue_current_primitive` 读出 |
| `cmem.h` | C 累加输入 SRAM。深度 = 4 个 subtile（一个 16×16 tile 拆 2×2=4 个 8×8 subtile）；FP22 内部精度。`write_fill_subtile(fmt, subtile, packets)`（fp8/fp16/fp32 → fp22 转换写入）；`read_subtile_fp22(id, out)`（给 first 模式的 c_bypass 用）；`subtile_valid` 标志位。 | `tensor_unit.cpp::write_input_d_to_cmem`（FillC 末尾批量写）；`current_compute_ready`（Compute 阶段检查就绪）；`issue_current_primitive`（读出 c bypass） |
| `dmem.h` | D 输出缓冲。结构与 CMem 对称：4 subtile × FP22。`write_subtile_fp22`（first 模式直接写）、`accumulate_subtile_fp22`（middle/tail 模式原位累加，调 `fadd_s1+s2`）、`read_subtile_fp22`、`dump_subtile_packets(fmt, subtile, out_packets)`（FP22→fp8/fp16/fp32 转换，准备 StoreD）。 | `tensor_unit.cpp::advance_compute`（从 retire 端写入或累加）；`build_d_store_packets`（StoreD 前 dump） |
| `meta_mem.h` | 稀疏元数据存储。仅 1 个 64 B packet，逻辑划 4×16 B line；`write_fill_packet(packet)`、`read_line(step_m, step_k, out16B)`。 | `tensor_unit.cpp::advance_fill_meta` 写入；`issue_current_primitive` 读出 16 B meta 喂给 `sparse_select` |
| `sparse_select.h` | 2:4 / 1:4 稀疏路由的纯函数实现。`route_sparse_2_4_primitive(meta, a_compact, b_dense, a_routed, b_routed)` 与 `route_sparse_1_4_primitive(...)` 把压缩 A payload 和 dense B 按 meta 还原回 8×8 primitive 拓扑；统一入口 `route_sparse_primitive(mode, ...)`。 | `tensor_unit.cpp::issue_current_primitive` 在 `sparsity_kind!=sparse_none` 时调用 |

### 1.3 控制 / 调试 / 配置（`tensor/open_tensorcore/tensor_control` 和 `tensor_helper`）

| 文件 | 主要功能 | 由谁调用 |
|---|---|---|
| `tensor_control/tc_decode.h` | 声明 `TcDecodedMmaCmd`（idesc + qualifier + operand_block 三方合并后的字段集）和 `TcDecodedLdStCmd`（占位）；`class TcDecode` 提供 `decode_tcu_mma` / `decode_tcu_ld_st` / `reset`。 | `tensor_unit.cpp::Impl::enqueue_tcu_mma` 持有 `tc_decode_` 实例并调用 |
| `tensor_control/tc_decode.cpp` | 实现 `decode_tcu_mma`：把 `rs1_value` memcpy 成 `idescriptor_t`、`core->lmem_read` 读 32 B `operand_block_t`、把 qualifier 7 位切成 6 个修饰位，合并产出 `TcDecodedMmaCmd`。内含 `kind_to_fmt`（idesc.kind→fmt_a/b/c/d 映射，覆盖 F16/BF16/TF32/F8F6F4/MXF8F6F4/F8F16/F16F8/I8）、`fmt_cd_to_fmt`、`sparsity_to_mode`。 | 同上 |
| `tensor_control/config_register.h` | 全局 `Config`（命令行驱动的仿真配置，独立测试用）和 **`TensorCoreMeta`**：贯穿计算流水的元数据（wgid/async_id/a_slot_id/b_slot_id/c_slot_id/c_subtile_id/in_prec/out_prec/c_prec/c_bypass_is_fp22/use_cmem_operand1/sparse_mode/sparse_meta[16]/valid）。全局实例 `g_cfg`。 | `tensor_compute/` 下绝大多数算子（cmem、tc_mul_pipe、tensor_core_top、tc_add4_pipe_fp22、tc_add_pipe）都 include 它读 `TensorCoreMeta` 与 `g_cfg` |
| `tensor_helper/tensor_debug_utils.h` | 声明 `op_string(TcuType, IntrTcuArgs) → {name, args_str}`：把 TCU 指令格式化成可读字符串。 | 调试/日志路径（如 `tensor_unit.cpp::dump_debug_state`） |
| `tensor_helper/tensor_debug_utils.cpp` | `op_string` 的实现，逐 `TcuType` switch 出名字 + 适当展开 args（如 `TMEM_CP.shape{N}.decomp{M}`）。 | 同上 |
| `tensor_helper/tensor_core_test_utils.h` | TensorCoreTop standalone 测试 helper：`load_inputs(sim, a, b, c)` 一键灌入 + `run(sim)` 单步 tick + 简单 reference matmul。**不进入主时序路径**。 | `tensor_helper/test/main.cpp`（独立单测，非 simx 主二进制） |
| `tensor_helper/tensor_mem_test_utils.h` | 本地 SRAM 测试 helper：`bulk_fill_tile_for_reference(amem/bmem/cmem, fmt, packets)` 一次性灌入完整 tile；用于参考路径与 standalone 测试。 | 同上 |
| `tensor_control/tcissue_unit.{h,cpp}` | ⚠ **legacy**。原 `TensorAsyncFrontend`：早期把 `mma_load/mma_store/wmma/tcu_mma` 拆分入队 `MemUop/PendingWmmaJob` 的实现。已被 `tensor_unit.cpp` 内部的 `MmaOp` 状态机替代。include 路径 `open_tensorcore/tensor_top/...` 已不存在。 | 无（不在 Makefile 中） |
| `tensor_control/tcwmma_retire_unit.{h,cpp}` | ⚠ **legacy**。原 WMMA 原语退休单元。 | 无 |
| `tensor_control/mem_controller.{h,cpp}` | ⚠ **legacy**。原 A/B/C/DMem 的状态机管理器（ready/pending/valid/wmma_pending/b_ws_locked 生命周期）。 | 无 |

---

## 2. TMEM 模块

> PTX `tcgen05` 专用 scratchpad，128 lane × 512 byte = 64 KB；内部 banked + swizzled；
> **不同于 LMEM**：LMEM 用 `Core::lmem_read/write` 普通字节访问；TMEM 只能通过 `tcgen05.ld/st/cp/mma/shift` 间接访问。

### 2.1 描述符与端口消息（`tensor/`，根目录）

| 文件 | 主要功能 | 由谁调用 |
|---|---|---|
| `idescriptor.h` | **PTX 对齐的 5 种描述符 POD 结构**：<br>① `idescriptor_t`（32 b，tcgen05.mma 指令描述符，含 kind/shape_m/n/sparsity/layout/transpose/saturate/negate 等位段）；<br>② `operand_block_t`（32 B，住 LMEM，含 a_taddr/d_taddr/b_sdesc/lanes_off/fmt_cd）；<br>③ `cpabulk_transfer_args_t`（32 B，住 LMEM，含 smem_addr/mbar_addr/coords[5]）；<br>④ `tensor_map_t`（128 B，住 DRAM，对齐 CUtensorMap）；<br>⑤ `mbarrier_state_t`（8 B，住 LMEM，phase[2]/pending_arrival[30]/pending_tx[12]/expected_arrival[20] 位封装）。<br>另含 `IDescriptorKind` 枚举、`CollectorState` 枚举。 | `tc_decode.cpp` 读 idesc+operand_block；`tma.cpp` 读 tensor_map+cpabulk_args；`core.cpp` mbarrier 镜像写 LMEM 时 |
| `tensor_mem_port_types.h` | TensorUnit / Tma ↔ TmemSystem 之间的 SimPort 消息：<br>① `TensorMemPortReq`（请求 ID + arbitration_age + Read/Write + `Tmem::PortRequestDesc` + write_packet）；<br>② `TensorMemPortRsp`（请求 ID + access_type + read_packet）；<br>③ `TensorAsyncOpCompletion`（仅 async_id，用于 TMA_WAIT / TC_FENCE 唤醒）。 | TensorUnit / Tma / TmemSystem / Core 都 include |

### 2.2 TMEM 内核（`tensor/`，根目录）

| 文件 | 主要功能 | 由谁调用 |
|---|---|---|
| `tmem.h` | `class Tmem` 公开 API：分配表（`alloc/free/seal_allocator/lookup_allocation/find_allocation_by_lane`）、字节读写（`handle_region_read_bytes/write_bytes/region_copy_in/out`）、packet 读写（`region_read_packet/write_packet`）、ready 标志（`set_payload_ready/set_meta_ready/set_meta_region`）+ visible 双缓冲（`publish_visible_state`）、端口预算与仲裁（`enqueue_port_request/arbitrate_requests/request_granted/consume_request_grant/refund_*`）、`region_shift_down`（tcgen05.shift 用）；容量常量（128 lanes × 512 cols = 64 KB；packet=64 B；默认 16 banks）；`TmemAllocation` 与 `TmemHandleBlockReason` 枚举；TADDR `0xFFFFFFFF` sentinel。 | `TmemSystem` 持有一个实例；`Core` 间接通过 `tmem_system_->...` 访问 |
| `tmem.cpp` | `Tmem` 全部实现：bank 数（从 `VORTEX_SIMX_TMEM_BANKS` 环境变量读，默认 16）+ 互素 swizzle 步长初始化、分配/释放（沿 payload_col_allocs 找空闲列段）、字节↔(bank,row,bank_byte) 映射、`region_shift_down`（一次性把 col_span 内所有 packet 按 row_bytes 偏移）、端口仲裁主循环（pending_read/write_requests deque 按 age + tag 排序，依 read/write_packet_budget 与每 bank 预算授权）。 | 被 `Tmem` 类成员函数调用；外部不直接使用 |
| `tmem_system.h` | `class TmemSystem : SimObject`：5 个 SimPort（`TensorExecuteReqIn/RspOut`、`CoreTransferReqIn/RspOut`、`AsyncOpCompletionOut`），转发分配/查询 API 到 `Tmem`，独立持有 `shift_transactions_`、`live_shift_busy_handles_`、`visible_shift_busy_handles_` 三套状态；声明 `issue_shift`、双缓冲 `publish_visible_state`、调试 `dump_debug_state`。 | `Core` 构造 `tmem_system_` 并绑端口（`core.cpp:94-100`） |
| `tmem_system.cpp` | `TmemSystem::tick()` 的实现：每周期① publish_visible_state；② tmem_.ensure_port_budgets；③ drain TensorExecuteReqIn + CoreTransferReqIn 各 1 条入 `Tmem`；④ advance_shift_engine 推进所有 `ShiftTransaction`（按 packet 同时发 read+write、最后调 `region_shift_down`）；⑤ `tmem_.arbitrate_requests`；⑥ complete_granted_port_requests 把响应 packet 推回 RspOut。`issue_shift` 注册新 transaction；shift 完成时推 AsyncOpCompletionOut。 | 同上 |
| `tmem_utils.h` | `Tmem` 的纯函数工具的接口声明：分两个 namespace —— `tmem_functional`（bank 数环境变量解析、互素 stride 选择、ceil_div、resize_storage/reset_storage、region_query、`logical_byte_to_physical` 等寻址工具）和 `tmem_timing`（`reset_port_budgets`、`consume_packet_ports`、`refund_packet_ports`）。 | `tmem.cpp` 内部 |
| `tmem_utils.cpp` | 上述函数的实现。最值得一看的是 `choose_coprime_stride`（找与 bank 数互素的步长，保证 swizzle 无冲突）和 `logical_byte_to_physical`（把 (logical_col, logical_line) 映射到 (bank, row, bank_byte)）。 | 同上 |
| `tmem_math_packet.h` | "数学行优先视图 ↔ 物理 packet 视图" 转换器接口：`TmemMathPacketLayoutKind`（LinearPacketStream / MathRowMajor / ALineNative / BLineNative / CSubtileNative 五种）、`TmemMathPacketLayout`（行/列/精度/tile 数 …）、`TmemMathPacketRegion`、`uses_math_packet_adapter`、`pack_math_packet`、`unpack_math_packet`、`packet_math_region`。 | 边界场景（参考流 / 测试 fixture），主时序路径目前不强制使用 |
| `tmem_math_packet.cpp` | 上述接口实现。包含 `fmt_bytes` / `ab_packet_count` / `cmem_packets_per_subtile` 等格式表，以及把 16×16 tile 在不同视图间 pack/unpack 的具体字节布局逻辑。 | 同上 |

---

## 3. TMA 模块

> 负责两类批量搬运：**DRAM ↔ LMEM**（`cp.async.bulk.tensor` LD/ST）与 **LMEM ↔ TMEM**（`tcgen05.cp` 的功能等价）。

| 文件 | 主要功能 | 由谁调用 |
|---|---|---|
| `tma.h` | `class Tma : SimObject`：3 个 SimPort（`TmemReqOut/TmemRspIn` 对接 TmemSystem.CoreTransferReq/Rsp；`AsyncOpCompletionOut` 对接 Core）+ 1 路 `CacheReqOut/CacheRspIn`（预留 cache timing 接口，当前未使用，Phase-2 还是走 `Core::dcache_read/write` 同步路径）。返回结构 `TmaCpAsyncBulkResult`（payload_size_bytes + tx_bound_mbar + tx_bytes）。私有 `PendingLmemToTmemCopyOp`（async_id/wid/col_base/col_span/lmem_addr/byte_offset/total_bytes/cursor/stage/packet 缓冲）；声明同步 API `cpabulk_tensor_load/store`、异步 API `issue_lmem_to_tmem_copy`、`is_lmem_addr` / `element_type_bytes` 静态工具、`dump_debug_state`。 | `Core` 构造 `tma_`；`Core::cpabulk_tensor_load/store` 与 `Core::launch_lmem_to_tmem_copy` 直接调 |
| `tma.cpp` | 全部实现。<br>① `cpabulk_tensor_load(tmap_addr, args_lmem_ptr, complete_tx)`：`dcache_read` 128 B `tensor_map_t` + `lmem_read` 32 B `cpabulk_transfer_args_t` + 按 64 B chunk 循环 `dcache_read(src) → lmem_write(dst)`；若 complete_tx=1 且 args.mbar_addr≠0 → 返回 `{tx_bound_mbar=args.mbar_addr, tx_bytes=total_bytes}`，让 Core 在 op 完成时调 mbarrier_complete_tx。<br>② `cpabulk_tensor_store`：方向反过来。<br>③ `issue_lmem_to_tmem_copy`：插入 `PendingLmemToTmemCopyOp` 到 unordered_map。<br>④ `advance_lmem_to_tmem_copy_ops()`（每 tick 推进）：按 packet 处理 —— 若 packet 全对齐（packet_offset=0 && packet_bytes=64）→ 直接 LMEM read 整 packet + 发 Write 请求（WaitWrite）；否则先发 Read（WaitRead）→ 拿到 packet → LMEM 合并子区间字节 → 发 Write（WaitWrite）。`cursor` 推到 total_bytes 时拉 `payload_ready=true` 并推 AsyncOpCompletionOut。<br>⑤ `element_type_bytes`：CUtensorMapDataType 编码 → 字节数。<br>⑥ `is_lmem_addr`：判断地址是否落在 LMEM 区间。 | `Core` 同步路径调 `cpabulk_tensor_load/store`；`Core::launch_lmem_to_tmem_copy` 调 `issue_lmem_to_tmem_copy`；SimPlatform tick 调 `Tma::tick → advance_lmem_to_tmem_copy_ops` |

---

## 4. mbarrier 异步执行

> mbarrier 对象**双重存在**：Core 侧 `MBarrierEntry`（按 LMEM 地址索引的 `unordered_map`，含 expected_arrival_count / pending_arrival_count / pending_tx_count / expected_tx_count / phase / waiters_bitmap）+ LMEM 中的 8 B `mbarrier_state_t`（每次 Core 改状态都通过 `mirror_mbarrier_to_lmem` 回写，让 kernel 能直接 load）。
>
> 文件不在 `tensor/` 下，但与 TMA / TensorUnit / Core 联动：实现在 `sim/simx/core.h` / `core.cpp`，描述符在 `tensor/idescriptor.h`。

| 函数（`core.cpp` 行号） | 主要功能 | 由谁调用 |
|---|---|---|
| `mbarrier_init` (1376) | 初始化 barrier：清状态，写 expected_arrival_count、pending_arrival_count=count、phase=0；清掉旧 waiter target；调 `mirror_mbarrier_to_lmem`。 | `execute.cpp::MBAR_INIT` (qualifier[0]=0) |
| `mbarrier_invalidate` (1392) | 销毁 barrier：先 resume 所有残留 waiter，从 map 删除，把 LMEM 8 B 清零。 | `execute.cpp::MBAR_INIT` (qualifier[0]=1) |
| `mbarrier_arrive` (1412) | `pending_arrival_count -= dec`；回写 LMEM；调 `try_complete_mbarrier`；返回当前 phase token（写到 rd 让 kernel 用作 wait token）。 | `execute.cpp::MBAR_ARRIVE` |
| `mbarrier_arrive_drop` (1431) | arrive + 永久减 expected_arrival_count；触发 `try_complete_mbarrier`。 | `MBAR_ARRIVE` (qualifier[1]=1) |
| `mbarrier_expect_tx` (1448) | `expected_tx_count += bytes`，回写 LMEM。 | `MBAR_EXPECT_TX` |
| `mbarrier_complete_tx` (1459) | `pending_tx_count += bytes`，回写 LMEM，调 `try_complete_mbarrier`。 | `MBAR_COMPLETE_TX`；以及 `Core::on_async_tensor_op_completed`（cp.async.bulk.tensor.mbarrier::complete_tx::bytes 路径） |
| `mbarrier_wait` (1472) | 阻塞等待：若 phase parity 已翻 → 清 waiter target 返回 true；否则把当前 warpgroup 全部 wid 写入 `mbarrier_wait_targets_` 并置 `waiters_bitmap`，返回 false → execute 层 suspend(WarpStallReason::MBarrier)。 | `MBAR_WAIT` |
| `mbarrier_test_wait` (1503) | 非阻塞 phase parity 查询，立即返回 ready bit。 | `MBAR_TEST_TRY_WAIT` (qualifier[0]=0) |
| `mbarrier_try_wait` (1515) | 带超时的阻塞等待。**当前 CModel 实现等价于 test_wait**（注释里标注 future work：补 wakeup at +(1<<bucket) cycles）。 | `MBAR_TEST_TRY_WAIT` (qualifier[0]=1) |
| `try_complete_mbarrier` (874) | 核心 phase advance：当 `pending_arrival==0 && pending_tx>=expected_tx` → 翻 phase 奇偶位、`pending_arrival = expected_arrival`、清空 tx 计数、回写 LMEM、`emulator_.resume(wid)` 唤醒所有 `waiters_bitmap` 中的 warp。 | 上述所有可能改 barrier 计数的函数末尾 |
| `mirror_mbarrier_to_lmem` (862) | 按 PTX §7.6.1 8 B 位布局打包 `MBarrierEntry` 并 `lmem_write` 到 mbar_addr。 | 所有改 barrier 状态的函数 |
| `on_async_tensor_op_completed` (903) | **异步完成的统一钩子**：(1) 若 `op.tx_bound_mbar!=0` → `mbarrier_complete_tx`；(2) `resume_async_waiters(op.async_id)` 唤醒 TMA_WAIT / TC_FENCE 等显式等待者；(3) `try_resume_fence_waiters`；(4) 若 op 被 `tcgen05.commit` 过（`op.committed=true`）→ `--pending_arrival_count` + 回写 + `try_complete_mbarrier`。 | `drain_tensor_execute_completion_notices`、`drain_tma_completion_notices` 各驱动；以及同步完成的 TMA load/store（直接置 completed=true 后立即调） |

---

## 5. tcgen05 兼容的 RISC-V 指令

### 5.1 Opcode

| Opcode 名 | 数值 | 用途 |
|---|---|---|
| `EXT2` / `RISCV_CUSTOM1` | `0x2B` | TMEM 管理 + cp.async.bulk.tensor |
| `EXT3` / `RISCV_CUSTOM2` | `0x5B` | tcgen05 sync + full mbarrier |
| `EXT4` / `RISCV_CUSTOM3` | `0x7B` | tcgen05 compute（mma / ld / st / wait） |

- C++ Opcode 枚举：`sim/simx/common/instr.h:48-51`
- RTL Opcode 编码：`hw/rtl/VX_gpu_pkg.sv:159-161`
- kernel 宏：`kernel/include/vx_intrinsics.h:35-37` 的 `RISCV_CUSTOM1/2/3`

### 5.2 funct3 → TcuType 全量

| Opcode/funct3 | TcuType | 主要功能 |
|---|---|---|
| EXT2/001 | TMEM_REL_PERMIT     | `tcgen05.relinquish_alloc_permit` |
| EXT2/010 | TMEM_ALLOC          | rd=TADDR, rs1=col_span, rs2 必须传 `0xffffffff` |
| EXT2/011 | TMEM_DEALLOC        | rs1=handle |
| EXT2/100 | TMEM_CP             | shared→TMEM 拷贝；rs1=taddr, rs2=LMEM ptr→s_desc；qualifier[3:1]=shape, [5:4]=decompress |
| EXT2/101 | TMEM_SHIFT          | `tcgen05.shift.down`；rs1=handle, rs2=control |
| EXT2/110 | CPABULK_TENSOR_LD   | DRAM→shared 批量；rs1=tensor_map ptr, rs2=args LMEM ptr；qualifier[5]=mbar_complete_tx |
| EXT2/111 | CPABULK_TENSOR_ST   | shared→DRAM 批量 |
| EXT3/000 | MBAR_FENCE          | `tcgen05.fence::{before,after}_thread_sync`；qualifier[0]=mode |
| EXT3/001 | MBAR_COMMIT         | `tcgen05.commit`；rs1=mbar_addr, rs2=cta_mask |
| EXT3/010 | MBAR_INIT           | qualifier[0]=init(0)/invalidate(1) |
| EXT3/011 | MBAR_ARRIVE         | qualifier[1]=drop, [2]=relaxed, [3]=expect_tx_combo |
| EXT3/100 | MBAR_EXPECT_TX      | rs2=tx_bytes |
| EXT3/101 | MBAR_COMPLETE_TX    | rs2=tx_bytes |
| EXT3/110 | MBAR_WAIT           | 阻塞等待；rs2=phase_token |
| EXT3/111 | MBAR_TEST_TRY_WAIT  | qualifier[0]=test(0)/try(1), [6:2]=timeout_bucket |
| EXT4/000 | TCU_MMA             | rs1=idesc (32 b), rs2=operand_block_t LMEM ptr，rd=x0 |
| EXT4/001 | TCU_LD              | rd=data, rs1=taddr，warp 协作（每线程 lane=taddr.lane+t） |
| EXT4/010 | TCU_ST              | rs1=taddr, rs2=value |
| EXT4/011 | TCU_WAIT_LD         | 等本 warp 在飞的 tcgen05.ld 完成 |
| EXT4/100 | TCU_WAIT_ST         | 等本 warp 在飞的 tcgen05.st 完成 |

### 5.3 文件分工

| 文件 | 主要功能 | 由谁调用 |
|---|---|---|
| `kernel/include/vx_tensor.h` | kernel 侧 inline asm 包装。20 余个 always_inline 函数 + 模板：`tmem_alloc / tmem_dealloc / tmem_rel_permit / tmem_cp / tmem_cp_shape / tmem_shift / cpabulk_tensor_ld / cpabulk_tensor_st`（EXT2）；`mbar_fence_before/after / mbar_commit / mbarrier_init / mbarrier_invalidate / mbarrier_arrive / mbarrier_arrive_token / mbarrier_arrive_drop / mbarrier_arrive_expect_tx / mbarrier_expect_tx / mbarrier_complete_tx / mbarrier_wait / mbarrier_test_wait / mbarrier_try_wait`（EXT3）；`tcu_mma / tcu_mma_no_accum / tcu_ld / tcu_st / tcu_wait_ld / tcu_wait_st`（EXT4）。还包含 host/device 共用构造器：`make_idescriptor<At,Bt,Ct,Dt,...>`、`make_operand_block`、`make_cpabulk_args`、`tmem_handle_base/span`。 | 用户 kernel 代码 |
| `kernel/include/vx_intrinsics.h` | 定义 `RISCV_CUSTOM0..3` 宏（值 `0x0B/0x2B/0x5B/0x7B`）。 | `vx_tensor.h` 引用 |
| `sim/simx/common/instr.h` | `enum class Opcode` 枚举的 EXT2/EXT3/EXT4 三项。 | `decode.cpp` |
| `sim/simx/common/types.h` | `enum class TcuType`（20 条）+ `enum class TcuFenceMode / TcuCpShape / TcuCpDecompress / TcuLdStShape / TcuTestTryWait` 子枚举；`IntrTcuArgs`（execute 阶段读的所有 qualifier 解码后字段，按 TCU 子族分块注释）；流式 `operator<<` 调试输出。 | `decode.cpp` / `execute.cpp` |
| `sim/simx/decode.cpp` 第 1111-1305 行 | EXT2/3/4 三个 case：切 `funct3` → 选 `TcuType`；切 `funct7` → 填 `IntrTcuArgs` 中对应位段；`make_tcu_instr(...)` 投递到 ibuffer；设置 rd/rs1/rs2。 | Core 的 decode 阶段 |
| `sim/simx/execute.cpp` 第 1458-1696 行 | 按 `TcuType` switch；分别调 `core_->tmem_*` / `core_->mbarrier_*` / `core_->cpabulk_tensor_*` / `core_->mbar_*` / `tensor_unit_->dispatch_tcu_mma` / `core_->issue_tcgen05_ld_async` / `tcgen05_st_async` / `tcgen05_wait_ld/st`。失败（retry / suspend）会 `core_->suspend(wid, ...)` 或 `set_stall_reason`。 | Core 的 execute 阶段 |
| `sim/simx/core.h / core.cpp` | 所有 tcgen05 控制面状态机的宿主：构造 `tensor_unit_` / `tmem_system_` / `tma_` 三个 SimObject 并绑端口；`AsyncTensorOp` 表（TmaLoad/Store/TmemCopy/TmemShift/MmaLoad/MmaStore/Wmma 共 7 种）；`PendingTcgen05LdStOp` 表（逐线程 taddr → packet 状态机）；`mbarriers_`、`mbarrier_wait_targets_`、`fence_wait_states_`、`tcgen05_ld/st_waiters_`、`async_tensor_waiters_`；`advance_async_tensor_engine`、`advance_tcgen05_ldst_async_ops`、`drain_*_completion_notices`、`on_async_tensor_op_completed`、`try_resume_fence_waiters`、`try_resume_tcgen05_ldst_waiters`、`try_complete_mbarrier`、`mirror_mbarrier_to_lmem`，外加全部 `tmem_*` / `mbarrier_*` / `mbar_*` / `cpabulk_tensor_*` / `tmem_cp / tmem_shift / tmem_dealloc / tmem_rel_permit / tmem_alloc / tmem_handle_*_block_reason` 等公开 API。 | `execute.cpp` 调用；SimPlatform 调度 `Core::tick` |

---

## 6. 端口拓扑（绑定在 `core.cpp:93-101`）

```
TensorUnit.TensorMemReqOut        ──▶ TmemSystem.TensorExecuteReqIn
TmemSystem.TensorExecuteRspOut    ──▶ TensorUnit.TensorMemRspIn
Tma.TmemReqOut                    ──▶ TmemSystem.CoreTransferReqIn
TmemSystem.CoreTransferRspOut     ──▶ Tma.TmemRspIn
TensorUnit.TensorAsyncOpCompletionOut ──▶ Core.tensor_async_op_completion_in_
Tma.AsyncOpCompletionOut              ──▶ Core.tma_async_op_completion_in_
TmemSystem.AsyncOpCompletionOut       ──▶ Core.tmem_system_async_op_completion_in_
```

---

## 7. 数据路径

按"数据存在哪 → 怎么搬 → 谁经手"梳理。

### 7.1 三大存储

| 存储 | 位置 | 谁拥有 | 谁能读/写 |
|---|---|---|---|
| **DRAM** | 全局内存，64 B chunk | 系统 | 普通 load/store；TMA 的 `dcache_read/write` |
| **LMEM**（Shared） | 每核 scratchpad | `Core::local_mem_` (LocalMem) | 普通 LSU 指令；`Core::lmem_read/write` 字节级；TMA / TensorUnit 读 PTX 描述符 / B 矩阵 |
| **TMEM** | 每核张量 scratchpad | `Core::tmem_system_` (TmemSystem 持有 Tmem) | 仅能通过 `tcgen05.ld/st/cp/mma/shift` 间接访问；端口：TensorExecute / CoreTransfer 两路 |

### 7.2 四条数据流

**a) DRAM ↔ LMEM**（`cp.async.bulk.tensor` LD/ST）
```
kernel.cpabulk_tensor_ld
  → execute.cpp::CPABULK_TENSOR_LD
    → Core::cpabulk_tensor_load
      → Tma::cpabulk_tensor_load                 // 同步实现
          ├─ dcache_read(tensor_map, 128 B)
          ├─ lmem_read(args, 32 B)
          └─ loop 64 B chunk: dcache_read(src) → lmem_write(dst)
```

**b) LMEM ↔ TMEM**（`tcgen05.cp` / TMA packet 路径）
```
Core::launch_lmem_to_tmem_copy
  → Tma::issue_lmem_to_tmem_copy → Tma::advance_lmem_to_tmem_copy_ops（每 tick）
      packet 状态机 Ready → WaitRead → WaitWrite：
        Tma.TmemReqOut ─▶ TmemSystem.CoreTransferReqIn
        TmemSystem ─▶ Tmem.region_read/write_packet（端口仲裁后）
        TmemSystem.CoreTransferRspOut ─▶ Tma.TmemRspIn
      op.cursor += 64 → 完成 → AsyncOpCompletionOut → Core 唤醒
```

**c) TMEM ↔ 计算阵列**（`tcgen05.mma` 一次宏指令内部展开）
```
            FillA / FillMeta / FillC                  StoreD
TMEM ◀─ packet read ─▶ AMem / MetaMem / CMem                    DMem ─ packet write ─▶ TMEM
                              │                                  ▲
                              ▼                                  │
                       TensorCoreTop(8×8) ── retire ─── DMem write/accum
                              │
            FillB             ▼
LMEM (b_sdesc 寻址) ── lmem_read(64 B per packet) ───────────▶ BMem
                              │
                         primitive A,B,C → FP22 D
```
搬运全部由 `tensor_unit.cpp::Impl` 的 `MmaOp` 状态机驱动；FillA/Meta/C/StoreD 走 TmemSystem，FillB 走 `Core::lmem_read` 直读，Compute 阶段只在本地 SRAM ↔ TensorCoreTop 之间走 PE 阵列。

**d) TMEM ↔ 寄存器堆**（`tcgen05.ld/st`，逐线程 warp 协作）
```
kernel.tcu_ld / tcu_st
  → execute.cpp::TCU_LD/ST
    → Core::issue_tcgen05_ld/st_async（注册 PendingTcgen05LdStOp）
      → Core::advance_tcgen05_ldst_async_ops（每 tick 推进）
          每线程 (taddr.lane + t, col_byte..+3) → packet_idx
          经 Tmem::enqueue_port_request 走 TmemSystem 仲裁
      → 全部 access 完成后 tcgen05_ld_trace_ready=true
        → commit 阶段把 4 字节写回 RF（或反向 RF→TMEM 完成 store）
```

---

## 8. 控制路径

按"指令进来 → 拆分 → 状态机 → 完成 → 唤醒"梳理。

### 8.1 前端控制（指令译码 + 派发）

```
kernel inline asm (vx_tensor.h)            // RISC-V .insn r ${opcode}, ${funct3}, ${funct7}, rd, rs1, rs2
            │
            ▼  32-bit instruction word
decode.cpp::Emulator::decode
   case Opcode::EXT2/3/4 (decode.cpp:1111-1305):
     - funct3 → 选 TcuType
     - funct7 → 填 IntrTcuArgs 各位段
     - make_tcu_instr → ibuffer
            │
            ▼  instr_trace_t
execute.cpp::Emulator::execute  (execute.cpp:1458-1696)
   switch (TcuType):
     TMEM_*            → core_->tmem_*
     CPABULK_*         → core_->cpabulk_tensor_*
     MBAR_*            → core_->mbarrier_* / core_->mbar_*
     TCU_MMA           → tensor_unit_->dispatch_tcu_mma     ← 唯一进 TensorUnit 的指令
     TCU_LD / TCU_ST   → core_->issue_tcgen05_ld/st_async
     *_WAIT / FENCE    → 若未就绪 → core_->suspend(wid, WarpStallReason::...)
```

### 8.2 TensorUnit 内部控制（`tcgen05.mma` 微展开）

```
TensorUnit::dispatch_tcu_mma → Impl::enqueue_tcu_mma:
  ├─ TcDecode::decode_tcu_mma                 // 解 idesc + lmem_read operand_block_t
  ├─ resolve_taddr(a_taddr) / (d_taddr)        // TADDR → (col_base, byte_offset)
  ├─ Core::tmem_handle_load/store_block_reason // 看 visible_payload_ready / shift_busy
  │     失败 → trace_data->retry=true → execute 层重发
  ├─ Core::wmma_async_issue → async_id
  └─ pending_mma_.push_back(MmaOp{stage=FillA, ...})

TensorUnit::tick:
  ├─ latch_tensor_instruction_pipe             // 普通 instr 流水延迟
  ├─ receive_tensor_mem_responses              // TmemSystem 响应入队
  └─ advance_mma(队首):
        FillA    → 发 packet read → AMem write_fill_line
        FillMeta → 发 packet read → MetaMem
        FillB    → Core::lmem_read → BMem write_fill_line
        FillC    → 发 packet read → CMem write_fill_subtile
        Compute  → push_uop → TensorCoreTop.tick → pop_retired → DMem write/accumulate
        StoreD   → 发 packet write → TMEM region
        Complete → TensorAsyncOpCompletionOut.push(async_id)
```

### 8.3 TMEM 端口仲裁控制（`TmemSystem::tick` 每周期 7 步）

```
1) publish_visible_state             // live → visible 双缓冲
2) tmem_.ensure_port_budgets          // 重置读/写端口预算
3) drain_one_inbound_request(TensorExecuteReqIn) → tmem_.enqueue_port_request
4) drain_one_inbound_request(CoreTransferReqIn)  → tmem_.enqueue_port_request
5) advance_shift_engine               // 推进 ShiftTransaction（一对 read+write per packet）
6) tmem_.arbitrate_requests           // 按 age + bank/port 预算授权
7) complete_granted_port_requests     // 实际读/写 TMEM → 回响应端口
```

### 8.4 异步完成与 mbarrier 汇合（控制路径的"出口"）

```
SimObject 通过 AsyncOpCompletionOut 推 TensorAsyncOpCompletion(async_id):
  TensorUnit ─▶ Core.tensor_async_op_completion_in_
  Tma        ─▶ Core.tma_async_op_completion_in_
  TmemSystem ─▶ Core.tmem_system_async_op_completion_in_

Core::tick → advance_async_tensor_engine:
  → drain_*_completion_notices → async_tensor_complete(async_id)
     → on_async_tensor_op_completed(op):                  // core.cpp:903
        (1) if op.tx_bound_mbar != 0:                     // cp.async.bulk.tensor.mbarrier::complete_tx
              mbarrier_complete_tx(addr, bytes)
        (2) resume_async_waiters(async_id)                // TMA_WAIT / TC_FENCE 显式等待者
        (3) try_resume_fence_waiters
        (4) if op.committed (被 tcgen05.commit 过):
              --pending_arrival_count + mirror + try_complete_mbarrier(addr)

try_complete_mbarrier (core.cpp:874):
  当 pending_arrival==0 && pending_tx>=expected_tx:
     phase 翻位 → pending_arrival = expected_arrival → 清 tx
     mirror_mbarrier_to_lmem                              // kernel 立即可见
     emulator_.resume(wid)  for each wid in waiters_bitmap
```

### 8.5 阻塞控制（warp suspend / resume）

| 阻塞场景 | 触发点 | 唤醒点 |
|---|---|---|
| `MBAR_WAIT` 未达 phase | `Core::mbarrier_wait` 返回 false → execute suspend (MBarrier) | `try_complete_mbarrier` → `emulator_.resume` |
| `MBAR_FENCE` 在飞 op 未完 | `Core::mbar_fence` 返回 false → suspend (AsyncTensor) | `try_resume_fence_waiters`（异步 op 完成时调） |
| `TCU_WAIT_LD/ST` 有未完成 ld/st | `Core::tcgen05_wait_ld/st` 返回 false → suspend (AsyncTensor) | `try_resume_tcgen05_ldst_waiters`（packet 全部完成时） |
| `TCU_MMA` 资源忙（队列满 / handle 冲突） | `dispatch_tcu_mma` 设 `trace_data->retry=true` → execute 层 `set_stall_reason(AsyncTensor)` | 下个周期重新发射，不挂起 warp |

---

## 附 A：构建源码白名单（`sim/simx/Makefile` 的 EXT_TCU_ENABLE 分支）

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

include path：
```
sim/simx        sim/simx/common       sim/simx/tensor
sim/simx/tensor/open_tensorcore
sim/simx/tensor/open_tensorcore/tensor_compute
sim/simx/tensor/open_tensorcore/tensor_control
sim/simx/tensor/open_tensorcore/tensor_helper
sim/common      hw
```

## 附 B：legacy 清单（不在构建中、include 路径已废弃，可跳过）

```
sim/simx/tensor/open_tensorcore/tensor_control/tcissue_unit.{h,cpp}      # 原 TensorAsyncFrontend
sim/simx/tensor/open_tensorcore/tensor_control/tcwmma_retire_unit.{h,cpp} # 原 WMMA retire unit
sim/simx/tensor/open_tensorcore/tensor_control/mem_controller.{h,cpp}    # 原 A/B/C/DMem 状态机
sim/simx/tensor/open_tensorcore/tensor_helper/legacy/                    # 历史 helper（空目录）
sim/simx/tensor/open_tensorcore/tensor_helper/test/                      # standalone test (main.cpp / otc_decode.cpp)
sim/simx/tensor/open_tensorcore/build/                                   # standalone build 临时输出
```
