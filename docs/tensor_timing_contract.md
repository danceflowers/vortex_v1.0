# Vortex SIMX Tensor Timing Contract

版本: v4.0

日期: 2026-03-30

绑定代码: `91a2d382c649dff7b27674130fc611c97cee218c`

适用对象: 当前工作区 [sim/simx](/mnt/d/wode_code_trunk/vortex/sim/simx) 中已经实现的 tensor 子系统，包括 `TMEM / TMA / TMEM_SHIFT / MMA_LOAD / WMMA / MMA_STORE / TC_COMMIT / TC_FENCE / TC_WAIT / TMA_WAIT / mbarrier`，以及直接影响这些对象时序可见性的 `Core / Scheduler / Scoreboard / TensorUnit / AMem / BMem / CMem / MetaMem`。

本文是**时序契约**，不是架构说明书。本文冻结的是:

1. 哪一拍允许 issue
2. 哪一拍进入哪一个硬件阶段
3. 哪一拍完成
4. 哪一拍状态可见
5. 哪一拍允许被下游消费
6. 同拍多请求时谁先走
7. 在无冲突条件下最短多少拍完成

本文不以 Cmodel 变量名定义硬件语义。正文优先从硬件行为角度描述，再给出当前 `simx` 中的对应实现字段和函数，帮助读者定位代码。

## 1. 目的与优先级

本文用于冻结当前 `simx` tensor 路径的可观察时序边界，作为后续 RTL / Cmodel 对齐的签核依据。

优先级规则如下:

1. 若本文与绑定 commit 下的代码不一致，以代码行为为准。
2. 若本文与设计报告冲突，以本文为准，因为本文只保留当前实现已经落地的 timing 行为。
3. 若代码里仍保留兼容旧路径但本文未列出，则该行为不属于本版本签核范围。

## 2. 阅读方法

对于第一次接触该 Cmodel 的读者，建议按以下顺序阅读:

1. 先看第 3 章术语，理解 `window / tile / packet / line / subtile / slot`。
2. 再看第 5 章对象与三层视图，理解数学矩阵、TMEM 逻辑视图、物理 bank 视图如何分离。
3. 再看第 6 章全局一拍内顺序，理解 `Core` 与 `TensorUnit` 谁先谁后。
4. 再看第 7 章资源与仲裁，理解为什么同一拍不是所有请求都能前进。
5. 再看第 8 章状态位与状态机，理解各类 `ready/pending/inflight` 到底表示什么。
6. 最后看第 9 章逐指令时序契约与第 10 章最短完成周期表。

## 3. 术语

### 3.1 硬件语义术语

1. `issue`
   指指令在 `execute()` 阶段被接受，开始创建或推动一个 tensor 事务。
2. `admit`
   指事务已经占用了某一级内部资源，例如进入 `async op` 表、进入 `mem_ops_`、进入 `pending_wmma_jobs_`。
3. `complete`
   指事务在内部状态机意义上完成，例如某个 async op 的所有 packet 传输与本地动作都结束。
4. `visible`
   指事务结果对应的状态位已经更新，并且后续流水可以观察到。
5. `consumable`
   指某状态不只“看得见”，而且已经满足下游启动条件，例如某 slot 已可被 `WMMA` 消费。
6. `retire`
   专指 primitive 从 TensorCore 流水尾部退出，并把结果合并到 `CMem(fp22)`。
7. `launch latency`
   指命令已经发出，但数据通路尚未真正开始搬运 packet 的前置固定延迟。对 `TMA_LOAD / TMA_STORE / TMEM_SHIFT refill` 尤其重要。
8. `packet transfer`
   指一次 `64B` 数据包在 `TMEM` 接口层前进一次。
9. `local port action`
   指本地 memory 端口在一拍内完成的一次读或写动作。

### 3.2 存储对象术语

1. `window`
   指 `TMEM` 中与某个数学矩阵片段对应的一块逻辑区域。
2. `tile`
   指数学矩阵语义下的 `16x16` 基本块。
3. `packet`
   指 `64B` 总线传输单位。`TMA`、`TMEM` 接口层、`MMA_LOAD/MMA_STORE` 都围绕 packet 推进。
4. `line`
   指 `AMem/BMem` 中的一条目标存储行，或 `TMEM` 逻辑视图中的一条逻辑行。除非特别说明，`line` 不表示“时序拍数”。
5. `subtile`
   指 `CMem` 中一个 `8x8` 的 C 子块。每个 `16x16` C tile 固定包含 4 个 subtile。
6. `physical row`
   指 `TMEM` 某个 bank 内的一行物理存储。

### 3.3 `beat` 一词的专门约定

为了避免“目标对象”和“时序拍数”混淆，本文统一采用以下约定:

1. `AMem/BMem` 的目标对象一律写作 `line`
2. `CMem` 的目标对象一律写作 `subtile`
3. `packet` 只表示 `64B` 总线传输单位
4. `cycle` 只表示全局时钟拍
5. 只有在描述总线或本地端口“一次传输/一次动作”时，才使用 `beat`

这里的 `beat` 在当前实现里的准确语义是:

1. 完成一个 tile 的本地 memory 落地或导出，所需的本地端口动作次数
2. 它不是目标 line/subtile 本身
3. 它也不是全局 cycle 数

例如:

1. `AMem::fill_lines() = 4` 表示“写完一个 A tile 的 4 条 `AMem line` 需要 4 次本地写动作”
2. `CMem::fill_subtiles(fp16) = 4` 表示“写完一个 C tile 的 4 个 `CMem subtile` 需要 4 次本地写动作”

### 3.4 实现映射术语

下列名词在正文中先按硬件语义使用，括号中给出当前 `simx` 对应实现。

1. `TMA launch latency`
   当前 Cmodel 对应 [core.cpp](/mnt/d/wode_code_trunk/vortex/sim/simx/core.cpp) 中 `AsyncTensorOp.remaining_launch_cycles`。
2. `A/B/C slot`
   当前 Cmodel 对应 [tensor_unit.cpp](/mnt/d/wode_code_trunk/vortex/sim/simx/tensor_unit.cpp) 中 `a_slots_ / b_slots_ / c_slots_` 三组独立数组。
3. `TMEM async op`
   当前 Cmodel 对应 [core.h](/mnt/d/wode_code_trunk/vortex/sim/simx/core.h) 中 `AsyncTensorOp` 和 [core.cpp](/mnt/d/wode_code_trunk/vortex/sim/simx/core.cpp) 中 `async_tensor_ops_`。
4. `MMA local transaction`
   当前 Cmodel 对应 [tensor_unit.cpp](/mnt/d/wode_code_trunk/vortex/sim/simx/tensor_unit.cpp) 中 `MemUop` 与 `mem_ops_`。
5. `WMMA job`
   当前 Cmodel 对应 `pending_wmma_jobs_ / active_wmma_job_ / pending_wmma_uops_`。
6. `async op issue cycle`
   当前 Cmodel 对应 [core.h](/mnt/d/wode_code_trunk/vortex/sim/simx/core.h) 中 `issue_cycle`。它记录该 async 事务在 `Core::execute()` 中被创建的全局 cycle。
7. `async op first eligible cycle`
   当前 Cmodel 对应 [core.h](/mnt/d/wode_code_trunk/vortex/sim/simx/core.h) 中 `first_service_cycle`。它记录该 async 事务最早允许被 `advance_async_tensor_engine()` 检查的全局 cycle。

## 4. 范围与排除项

本版本签核范围只覆盖以下主路径:

1. descriptor-based `TMA_LOAD / TMA_STORE / TMEM_SHIFT`
2. descriptor-based `MMA_LOAD / MMA_STORE`
3. 宏 `WMMA` 进入 `TensorUnit` 后拆成 8 条 primitive
4. `TMEM = 64 columns x 128 lines` 逻辑视图
5. `TMEM = 16 banks x 64 rows x 64b` 物理视图
6. `CMem = fp22` 存储与累加

本版本明确排除以下旧路径:

1. 旧版 non-descriptor `MMA_LOAD / MMA_STORE`
2. 旧的 tagged-handle `window_id`
3. 未绑定 shaped window 的 legacy linear packet 路径
4. 尚未重新收敛的 sparse compute 前端行为

## 5. 对象与地址分层

### 5.1 数学矩阵视图

`A_shape / B_shape / C_shape / D_shape` 表示数学矩阵元素维度，不表示字节数，也不表示 TMEM 地址。

例:

1. `A_shape = 16 x 64` 表示 16 行 64 列元素。
2. `fmt` 只改变每个元素的字节数，不改变数学 shape。

### 5.2 TMEM 逻辑视图

当前 `TMEM` 逻辑视图为:

1. `64 columns`
2. `128 lines`

`window planner` 根据:

1. 数学矩阵 shape
2. 数据精度 `fmt`
3. 模式信息 `target / sparse / transpose`

生成:

1. `logical_col_base`
2. `logical_line_base`
3. `logical_col_span`
4. `logical_line_span`
5. `tile_count`
6. `packets_per_tile`

### 5.3 物理 bank 视图

当前 `TMEM` 物理阵列为:

1. `16 banks`
2. 每个 bank `64 rows`
3. 每行 `64b = 8B`

物理视图只回答:

1. 这一拍 packet 是否能通过 ingress/egress
2. 这一拍相关 bank 是否有端口
3. logical `(column, line)` 如何落到 physical `(bank, row, byte)`

### 5.4 三层视图之间的职责边界

必须严格分开理解:

1. 数学矩阵视图决定 `tile/packet` 的语义与顺序。
2. `TMEM` 逻辑视图决定 window footprint。
3. 物理 bank 视图决定带宽与仲裁。

换句话说:

1. 上层看到的是数学 tile 和数学 packet。
2. 中层看到的是 `window -> (column, line)`。
3. 底层看到的是 `(bank, row, byte)`。

## 6. 全局时间基准与平台级顺序

### 6.1 全局 cycle

整个 tensor 子系统与普通流水共享同一个全局 cycle:

1. [core.cpp](/mnt/d/wode_code_trunk/vortex/sim/simx/core.cpp) 中 `perf_stats_.cycles`

不存在第二个 clock domain，不存在独立 tensor scheduler 时钟。

### 6.2 Core 内部顺序

当前 [core.cpp](/mnt/d/wode_code_trunk/vortex/sim/simx/core.cpp) 中 `Core::tick()` 的冻结顺序为:

1. `advance_async_tensor_engine()`
2. `commit()`
3. `execute()`
4. `issue()`
5. `decode()`
6. `fetch()`
7. `schedule()`
8. `cycles++`

这决定了:

1. 在 `advance_async_tensor_engine()` 中完成的 `TMA_LOAD / TMA_STORE / TMEM_SHIFT`，其 `ready/epoch` 变化对同拍 `execute()/issue()/schedule()` 可见。
2. 在 `execute()` 中新 issue 的普通 tensor 指令，本拍已经完成了 `advance_async_tensor_engine()`，因此不会在同一 `Core::tick()` 内再次触发 async packet 推进。

### 6.3 TensorUnit 内部顺序

当前 [tensor_unit.cpp](/mnt/d/wode_code_trunk/vortex/sim/simx/tensor_unit.cpp) 中 `TensorUnit::tick()` 的冻结顺序为:

1. `advance_tensor_memory_pipeline()`
2. `issue_wmma_primitives()`
3. `sample_pending_wmma_depth()`
4. `advance_tensorcore_pipeline()`

这决定了:

1. 本拍 `advance_tensor_memory_pipeline()` 看不到本拍稍后 `advance_tensorcore_pipeline()` 刚退休的 primitive。
2. 本拍 `issue_wmma_primitives()` 也看不到本拍稍后 `advance_tensorcore_pipeline()` 刚更新的 `CMem`。
3. 因此“最后一个 primitive retire 同拍立刻启动 `MMA_STORE`”不成立，最早下一拍才能启动。

### 6.4 平台级对象顺序

当前 `simx` 由 [simobject.h](/mnt/d/wode_code_trunk/vortex/sim/common/simobject.h) 中 `SimPlatform::tick()` 统一驱动所有 `SimObject`。对象按创建顺序执行 `do_tick()`。

结合 [simobject.h](/mnt/d/wode_code_trunk/vortex/sim/common/simobject.h) 中 `create_object()` 的语义，以及 [core.cpp](/mnt/d/wode_code_trunk/vortex/sim/simx/core.cpp) 中 `Core` 构造期间创建 `TensorUnit` 的事实，当前平台级顺序冻结为:

1. `TensorUnit::tick()` 先于同周期的 `Core::tick()`

这条顺序非常关键，因为它决定:

1. 本拍 `Core::execute()` 新 issue 的 `MMA_LOAD / MMA_STORE / WMMA`，**最早下一全局拍**才会被 `TensorUnit::tick()` 观察到。
2. 本拍 `TensorUnit::tick()` 刚刚完成的 `MMA_LOAD complete / WMMA retire / MMA_STORE complete`，在稍后的同拍 `Core::tick()` 中已经可被 `execute()/issue()/schedule()` 观察到。
3. 因此 `MMA_LOAD complete -> WMMA issue` 可以发生在**同一全局拍**，但前提是 `MMA_LOAD` 的 complete 发生在当拍早些时候的 `TensorUnit::tick()`。
4. 相反，`execute()` 新建的 `TMA_LOAD / TMA_STORE / TMEM_SHIFT async op` 虽然本拍建立，但真正的 async 推进最早仍要到下一全局拍的 `advance_async_tensor_engine()`。

## 7. 资源预算与仲裁规则

### 7.1 TMEM 接口层预算

当前 TMEM 对外接口层冻结为:

| 方向 | 预算 | 说明 |
| --- | --- | --- |
| write ingress | 1 packet/cycle | `TMA_LOAD`、`TMEM_SHIFT refill`、其他向 TMEM 写回的路径共享 |
| read egress | 1 packet/cycle | `TMA_STORE`、`MMA_LOAD`、`TMEM_SHIFT` 内部读出共享 |

这意味着:

1. `bank` 不冲突，不代表就能同拍通过多个 packet。
2. 先受共享 ingress/egress 限制，再受 bank 限制。

### 7.2 TMEM bank 仲裁

每个 `64B packet` 在物理映射层被拆成 `8` 个 `8B lane`，通过 swizzle 打散到 physical bank。

bank 层冻结规则:

1. 一个 packet 只有在其涉及的所有 bank 都有对应方向的端口预算时，才允许前进。
2. 每个 bank 当前为 `1R1W`。
3. bank 层不改变对外 `1 packet/cycle` 的接口限制。

### 7.3 Core 侧 async op 仲裁

当前 `Core::advance_async_tensor_engine()` 先统一重置本拍 `TMEM` 端口预算，再按独立 FIFO 队列 `async_tmem_ops_fifo_` 的 issue 顺序推进 `TMA_LOAD / TMA_STORE / TMEM_SHIFT`。

冻结规则:

1. 每个 async op 在本拍至多被 `advance_one_async_tensor_transaction()` 检查一次。
2. 只有 `TMA_LOAD / TMA_STORE / TMEM_SHIFT` 进入这条 FIFO；`MmaLoad / MmaStore / Wmma` 仍只保留在 `async_tensor_ops_` 中用于 waiter / barrier 跟踪，不参与 `Core` 侧 TMEM 口竞争。
3. 若队列中更早的 async op 在本拍请求 `TMEM` 口但因共享 ingress/egress 或 bank 端口不足而失败，则本拍停止继续检查更年轻的 async op。
4. 因此“多个独立 async op 同拍谁先抢到 `TMEM` 口”现在按 issue 顺序冻结为严格 FIFO。
5. 仍然允许更早的 async op 在本拍只消耗 launch latency、而不请求 `TMEM` 口；这种情况下后面的 async op 仍可继续被检查。

### 7.4 本地 memory 端口预算

当前 `TensorUnit` 本地端口预算按拍重置，冻结为:

1. `AMem write = 1 local port action / cycle`
2. `BMem write = 1 local port action / cycle`
3. `CMem write = 1 local port action / cycle`
4. `CMem read = 1 local port action / cycle`
5. `MetaMem write = 1 local port action / cycle`

### 7.5 TensorUnit 内部 `mem_ops_` 仲裁

`mem_ops_` 是 `deque`，保存 `FillA / FillB / FillC / StoreC`。

冻结规则:

1. `advance_tensor_memory_pipeline()` 每拍从队首开始顺序扫描。
2. 较早入队的 `MemUop` 拥有更高优先级。
3. 同拍内可以有多个 `MemUop` 获得“部分推进”，但一旦某个 `MemUop` complete 并被弹出，`advance_tensor_memory_pipeline()` 立即返回。
4. 因此每拍至多有一个 `MemUop` 真正 complete 并出队。

### 7.6 单个 `MemUop` 内部优先级

#### 7.6.1 FillA / FillB / FillC

当前单个 fill uop 的本拍动作优先级冻结为:

1. 先尝试本地写动作
2. 只有当前没有可执行的本地写动作时，才尝试新的 TMEM packet 读取

细化为:

1. `FillA`: `AMem line write` 优先于 `MetaMem write`，`MetaMem write` 优先于新的 `TMEM read`
2. `FillB`: `BMem line write` 优先于新的 `TMEM read`
3. `FillC`: `CMem subtile write` 优先于新的 `TMEM read`

这意味着:

1. 本拍一旦已经具备本地写条件，就不会在同一 `MemUop` 上继续读取新的 TMEM packet。
2. 同一 `MemUop` 内部的 packet 读取和本地写入是串行交替推进的。

#### 7.6.2 StoreC

当前单个 `StoreC` 的本拍动作优先级冻结为:

1. 若 staged store packet 已经存在，则优先尝试 `TMEM write`
2. 只有 staged store packet 已全部写完且仍有剩余 `CMem` subtile 待读时，才尝试新的 `CMem read`

因此:

1. `StoreC` 的局部顺序是“先读两块 subtile 组成一组，再把这组对应的 packet 全部写完，然后继续读下一组 subtile”。
2. 不允许在 staged packet 尚未 drain 完时继续预取更多 subtile。

### 7.7 WMMA job 仲裁

当前 `WMMA` job 队列冻结为:

1. `pending_wmma_jobs_` 是 FIFO `deque`
2. `active_wmma_job_` 为空时，`issue_wmma_primitives()` 才会取队首 job 成为 active
3. 一旦成为 `active_wmma_job_`，该 job 会一直保留到 8 条 primitive 全部发射完
4. 当前不存在多个 job 的 primitive 交错发射

### 7.8 Scheduler 的最小 contract

当前 tensor-aware 调度至少满足:

1. 明知本拍一定 retry 的 tensor warp 不应获得高优先级
2. `WMMA > MMA_LOAD > MMA_STORE`
3. `handle busy` 与 `slot busy` 必须分开判定
4. `payload_ready / meta_ready / cmem_final_valid` 属于 issue 前阻塞条件，而不是“先发出去再靠 execute 重试”

## 8. 状态、计数器与状态机

### 8.1 TmemAllocation 状态

当前与时序直接相关的 allocation 状态包括:

1. `valid`
2. `payload_ready`
3. `meta_ready`
4. `layout_valid`
5. `layout_epoch`
6. `windows`

硬件语义:

1. `payload_ready=false` 表示该 allocation 对应 payload window 的内容正在被写入或正在被 shift，暂不允许被消费。
2. `meta_ready=false` 表示相应 meta shadow window 仍未就绪。
3. `layout_epoch` 表示“这个 window 内容版本号是否发生变化”。任何依赖旧内容的 cursor 若不校验 epoch，就可能读到脏内容。

实现映射:

1. 当前由 [tmem.h](/mnt/d/wode_code_trunk/vortex/sim/simx/tmem.h) / [tmem.cpp](/mnt/d/wode_code_trunk/vortex/sim/simx/tmem.cpp) 维护。

### 8.2 Slot 状态

当前与时序直接相关的 slot 状态包括:

1. `a_ready / b_ready / c_ready`
2. `a_pending / b_pending / c_pending`
3. `wmma_pending`
4. `c_wmma_inflight`
5. `c_dirty`
6. `cmem_final_valid`
7. `store_pending`

#### 8.2.1 “对应 slot” 的准确含义

本文中“对应 slot”一词，必须按指令类型展开理解，不能模糊地理解成“某个统一 slot 结构”。

当前冻结映射为:

| 指令/target | 实际操作的 slot 数组 | slot 索引来源 |
| --- | --- | --- |
| `MMA_LOAD(target=A)` | `a_slots_[slot_id]` | `args.slot_id` |
| `MMA_LOAD(target=B)` | `b_slots_[slot_id]` | `args.slot_id` |
| `MMA_LOAD(target=C)` | `c_slots_[slot_id]` | `args.slot_id` |
| `MMA_STORE` | `c_slots_[slot_id]` | `args.slot_id` |
| `WMMA` | `a_slots_[a_slot_id]`、`b_slots_[b_slot_id]`、`c_slots_[c_slot_id]` | `args.a_slot_id / b_slot_id / c_slot_id` |

补充说明:

1. `A slot0`、`B slot0`、`C slot0` 不是同一个对象，而是三组不同数组中的第 0 项。
2. 只有在 `WMMA` 里，三组 slot 才会同时组合起来使用。
3. 因此文中类似“对应 slot 的 `*_pending=true`”都必须按 target 展开理解。

#### 8.2.2 Slot 位的硬件语义

1. `*_pending`
   表示这个 slot 当前正被本地 fill/store 事务占用，内容尚未完整落地。
2. `*_ready`
   表示这个 slot 已完成当前语义所需的完整本地写入，可以作为下游输入使用。
3. `wmma_pending`
   表示该 A/B slot 已经被某个已入队但尚未完成 primitive 发射的 `WMMA` 宏操作占用。
4. `c_wmma_inflight`
   表示仍有尚未 retire 的 primitive 结果最终会写入该 C slot。
5. `c_dirty`
   表示 C 数据已经被修改，尚未由 `MMA_STORE` 导出到 TMEM。
6. `cmem_final_valid`
   表示该 C slot 里的结果当前可作为 `MMA_STORE` 的源。
7. `store_pending`
   表示该 C slot 正在进行 `MMA_STORE`。

#### 8.2.3 Slot 位的典型置位/清零条件

下表描述冻结的硬件语义。最后一列给出当前 `simx` 的主要实现入口，帮助定位代码，但不替代前两列的定义。

| 状态位 | 置位时机 | 清零时机 | 当前实现映射 |
| --- | --- | --- | --- |
| `a_pending` | `MMA_LOAD(target=A)` issue 并成功建立 `FillA` 时 | `FillA` complete 时 | `mark_a_pending()/mark_a_ready()` |
| `b_pending` | `MMA_LOAD(target=B)` issue 并成功建立 `FillB` 时 | `FillB` complete 时 | `mark_b_pending()/mark_b_ready()` |
| `c_pending` | `MMA_LOAD(target=C)` issue 并成功建立 `FillC` 时 | `FillC` complete 时 | `mark_c_pending()/mark_c_ready()` |
| `a_ready` | `FillA` 全部目标 line 落地时 | 新 `FillA` issue，或该 slot 被 active `WMMA` 完成 8 条 primitive 发射后 | `mark_a_ready()/issue_wmma_primitives()` |
| `b_ready` | `FillB` 全部目标 line 落地时 | 新 `FillB` issue，或该 slot 被 active `WMMA` 完成 8 条 primitive 发射后 | `mark_b_ready()/issue_wmma_primitives()` |
| `c_ready` | `FillC` 全部 4 个 `CMem subtile` 落地时 | 新 `FillC` issue 时 | `mark_c_ready()/mark_c_pending()` |
| `wmma_pending` | `WMMA` 宏操作入队时，对应 A/B slot 置位 | 对应 active job 完成第 8 条 primitive 发射时 | `enqueue_async_wmma()/issue_wmma_primitives()` |
| `c_wmma_inflight` | `WMMA` 宏操作入队时加 1 | 对应宏操作最后一条 primitive retire 时减 1 | `enqueue_async_wmma()/retire_primitive()` |
| `c_dirty` | 该 C slot 的任一 `WMMA` 完成最终 retire 时置位 | `FillC` 对该 slot 完整覆盖时清零 | `retire_primitive()/mark_c_ready()` |
| `cmem_final_valid` | `FillC` complete，或该 slot 最后一个 primitive retire 时 | 新 `FillC` issue，或 `WMMA` issue 时 | `mark_c_ready()/mark_c_pending()/enqueue_async_wmma()/retire_primitive()` |
| `store_pending` | `MMA_STORE` issue 并成功建立 `StoreC` 时 | `StoreC` complete 时 | `enqueue_async_mma_store()/advance_tensor_memory_pipeline()` |

### 8.3 本地 memory 的细粒度 valid

除了 slot 聚合位之外，本地 memory 容器内部还维护更细粒度的 valid 状态。它们同样属于 timing contract。

#### 8.3.1 AMem

当前 `AMem` 维护:

1. 共享物理存储
2. `slot0/slot1` 固定深度窗口
3. `row_valid_[8]`

硬件语义:

1. `row_valid_[slot_base + line] = true` 表示该 `A line` 已写入完毕，物理上可被 `read_primitive()` 读取。
2. `row_valid_` 是逐 line 置位，不是整 slot 一次性置位。

#### 8.3.2 BMem

`BMem` 与 `AMem` 同理，也维护逐 line `row_valid_`。

#### 8.3.3 CMem

当前 `CMem` 维护:

1. 共享物理存储
2. 每个 slot 4 条 `8x8` subtile row
3. `row_valid_[8]`

硬件语义:

1. `row_valid_[slot_base + subtile] = true` 表示该 `8x8` subtile 已存在可读 `fp22` 数据。
2. `write_fill_subtile()` 逐 subtile 置位。
3. `accumulate_subtile()` 要求目标 subtile 已 valid。

#### 8.3.4 MetaMem

当前 `MetaMem` 维护:

1. 每个 slot 1 个 `64B packet`
2. `valid_[slot]`

硬件语义:

1. `valid_[slot]=true` 表示整个 meta packet 已到位。
2. 虽然 `read_line(step_m, step_k)` 按 `4 x 16B` line 读取，但 valid 判定仍然是 slot 级 packet valid。

### 8.4 聚合 ready 与细粒度 valid 的关系

当前实现不是 fully line-level ready 调度，而是**slot 级控制 + line/subtile 级存储 valid** 的混合模型。

冻结规则:

1. `a_ready/b_ready/c_ready` 是 slot 级聚合位，不是 `row_valid_` 的实时组合逻辑输出。
2. `a_ready/b_ready/c_ready` 只在对应 `MemUop` 全部完成时，由 `mark_a_ready()/mark_b_ready()/mark_c_ready()` 置位。
3. 因此“某几条 line 已 valid 就允许提前发 `WMMA`”在当前实现中不成立。
4. `cmem_final_valid` 是 slot 级聚合位，不是 `4` 个 subtile final 位的显式归约结果。
5. `cmem_final_valid` 在 `WMMA` issue 时清零，在最后一个相关 primitive retire 时重新置位。
6. `c_dirty` 也是 slot 级位，不是显式的 `c_subtile_dirty[4]`。

### 8.5 `remaining_*` 计数器的统一解释

为了防止读者把它们误读成“软件变量”，本文统一从硬件剩余工作量的角度解释这些计数器。当前 Cmodel 只是用 `remaining_*` 变量来实现这些硬件阶段。

#### 8.5.1 `remaining_launch_cycles`

硬件语义:

1. 表示命令已经发出，但数据通路尚未真正开始 packet 传输的启动延迟。
2. 对 `TMA_LOAD / TMA_STORE`，这对应 TMA engine 的命令解析、地址准备、启动握手等前置延迟。
3. 对 `TMEM_SHIFT refill`，这对应 refill 用 TMA 读回顶层数学行之前的启动延迟。

实现映射:

1. [core.cpp](/mnt/d/wode_code_trunk/vortex/sim/simx/core.cpp) 中 `AsyncTensorOp.remaining_launch_cycles`
2. 在 `advance_one_async_tensor_transaction()` 开头每拍减 1，归零前不做 packet 传输

#### 8.5.2 `remaining_tmem_read_packets / remaining_tmem_write_packets`

硬件语义:

1. 表示该事务还剩多少个 `TMEM` packet 读/写动作未完成。
2. 适用于 `TMA_LOAD / TMA_STORE / TMEM_SHIFT / MMA_LOAD / MMA_STORE`。

#### 8.5.3 `remaining_amem_fill_lines / remaining_bmem_fill_lines`

硬件语义:

1. 表示 `AMem/BMem` 还剩多少条目标 line 尚未写入完成。
2. 当前每个 tile 固定是 4 条 line。

#### 8.5.4 `remaining_cmem_fill_subtiles / remaining_cmem_dump_subtiles`

硬件语义:

1. `remaining_cmem_fill_subtiles` 表示 `CMem` 还剩多少个 subtile 尚未从 `MMA_LOAD(C)` 落地。
2. `remaining_cmem_dump_subtiles` 表示 `MMA_STORE` 还剩多少个 subtile 尚未从 `CMem` 读出。

#### 8.5.5 `remaining_metamem_fill_packets`

硬件语义:

1. 表示 `MetaMem` 还剩多少个 meta packet 尚未写入。
2. 当前每个 A sparse tile 固定为 1 个 meta packet。

#### 8.5.6 `issue_cycle`、`first_service_cycle` 与 cursor

这些字段不是“调试用变量”，而是时序契约的一部分，因为它们决定了事务最早从哪一拍开始推进、以及当前已经推进到第几个 packet。

1. `issue_cycle`
   表示该 async op 在 `Core::execute()` 中被建立的全局 cycle。
2. `first_service_cycle`
   表示该 async op 最早允许被 `advance_async_tensor_engine()` 检查的全局 cycle。当前 `TMA_LOAD/TMA_STORE/TMEM_SHIFT` 都设置为 `issue_cycle + 1`，因此它们不会在 issue 当拍就开始 packet 传输。
3. `next_payload_packet_idx`
   表示下一次 payload packet 传输将处理的 packet 序号。
4. `next_meta_packet_idx`
   表示下一次 metadata packet 传输将处理的 packet 序号。
5. `next_refill_line_packet_idx`
   表示 `TMEM_SHIFT refill` 顶部数学行回填下一次将处理的 line packet 序号。

#### 8.5.7 `remaining_*` 统一表

| 字段 | 硬件语义 | 适用对象 | 何时初始化 | 每拍何时减 1 | 归零后表示什么 |
| --- | --- | --- | --- | --- | --- |
| `remaining_launch_cycles` | 启动延迟尚未消耗完 | `TMA_LOAD/TMA_STORE/TMEM_SHIFT refill` | async op admit 时 | `advance_one_async_tensor_transaction()` 开头 | 可以进入 packet 级传输 |
| `remaining_tmem_read_packets` | 从 `TMEM` 读出的 packet 还剩多少个 | `TMA_STORE/TMEM_SHIFT/MMA_LOAD` | admit 时按 window/tile/format 计算 | 成功读到 1 个 packet 时 | 该事务不再需要 `TMEM read` |
| `remaining_tmem_write_packets` | 写入 `TMEM` 的 packet 还剩多少个 | `TMA_LOAD/TMEM_SHIFT refill/MMA_STORE` | admit 时按 window/tile/format 计算 | 成功写入 1 个 packet 时 | 该事务不再需要 `TMEM write` |
| `remaining_amem_fill_lines` | `AMem` 还剩多少条目标 line 未落地 | `FillA` | 建立 `FillA` 时固定为 4 | 成功执行 1 次 `AMem line write` 时 | `A tile` 的 4 条 line 全部到位 |
| `remaining_bmem_fill_lines` | `BMem` 还剩多少条目标 line 未落地 | `FillB` | 建立 `FillB` 时固定为 4 | 成功执行 1 次 `BMem line write` 时 | `B tile` 的 4 条 line 全部到位 |
| `remaining_cmem_fill_subtiles` | `CMem` 还剩多少个 subtile 未落地 | `FillC` | 建立 `FillC` 时固定为 4 | 成功执行 1 次 `CMem subtile write` 时 | `C tile` 的 4 个 subtile 全部到位 |
| `remaining_cmem_dump_subtiles` | `MMA_STORE` 还剩多少个 subtile 未从 `CMem` 读出 | `StoreC` | 建立 `StoreC` 时固定为 4 | 成功执行 1 次 `CMem subtile read` 时 | 所有 subtile 都已被读出并可组包 |
| `remaining_metamem_fill_packets` | `MetaMem` 还剩多少个 meta packet 未落地 | `FillA sparse` | 建立 `FillA sparse` 时固定为 1 | 成功执行 1 次 `MetaMem packet write` 时 | meta packet 已全部到位 |

#### 8.5.8 不同精度下 `remaining_*` 初值

| 事务 | `fp8` | `fp16` | `fp32` | 说明 |
| --- | --- | --- | --- | --- |
| `FillA` `remaining_tmem_reads` | 4 | 8 | 不适用 | 若 `A sparse`，再额外读取 1 个 meta packet |
| `FillB` `remaining_tmem_reads` | 4 | 8 | 不适用 | 当前 `B` 主路径按 dense 解释 |
| `FillC` `remaining_tmem_reads` | 4 | 8 | 16 | `remaining_cmem_fill_subtiles` 始终固定为 4 |
| `StoreC` `remaining_tmem_writes` | 4 | 8 | 16 | `remaining_cmem_dump_subtiles` 始终固定为 4 |
| `FillA/FillB` 本地目标数 | 4 lines | 4 lines | 不适用 | 目标 line 数固定，与精度无关 |
| `FillC` 本地目标数 | 4 subtiles | 4 subtiles | 4 subtiles | 目标 subtile 数固定，与精度无关 |
| `StoreC` 本地读取数 | 4 subtiles | 4 subtiles | 4 subtiles | 源 subtile 数固定，与精度无关 |

### 8.6 `MemUop` 详细状态机

`MemUop` 是 `MMA_LOAD` 和 `MMA_STORE` 在 `TensorUnit` 内部的本地搬运事务描述符。它不是 ISA 级指令，而是“本地 tile 搬运过程”的状态机。

#### 8.6.1 通用状态

1. `Admitted`
   已经进入 `mem_ops_`
2. `FetchFromTMEM`
   正在从 `TMEM` 读取 packet
3. `PacketStaging`
   packet 已到达 staged buffer，等待满足本地写入或后续组合条件
4. `LocalWrite`
   正在向 `AMem/BMem/CMem/MetaMem` 执行本地写动作
5. `LocalRead`
   仅 `StoreC` 使用，表示正在从 `CMem` 读取 subtile
6. `StorePacketAssemble`
   仅 `StoreC` 使用，表示两个 subtile 已组成 staged store packet
7. `WriteBackToTMEM`
   仅 `StoreC` 使用，表示正在把 staged packet 写回 `TMEM`
8. `Completed`
   所有剩余计数器归零，并且对应聚合 slot 位已更新

#### 8.6.1A 通用状态转移表

| 当前状态 | 进入条件 | 本拍允许动作 | 退出条件 | 下一状态 |
| --- | --- | --- | --- | --- |
| `Admitted` | `MMA_LOAD/MMA_STORE` issue 成功并建立 `MemUop` | 无；等待下一全局拍进入 `TensorUnit::tick()` | 被 `advance_tensor_memory_pipeline()` 扫描到 | `FetchFromTMEM` 或 `LocalRead` |
| `FetchFromTMEM` | `remaining_tmem_reads != 0` | 从 `TMEM` 读取 1 个 payload 或 meta packet 到 staged buffer | staged 数据足够支撑 1 次本地写动作，或本拍端口不足 | `PacketStaging` |
| `PacketStaging` | staged buffer 中已有数据 | 检查当前 staged 数据能否形成 1 次本地写动作或 1 组 store packet | 条件满足 | `LocalWrite` 或 `StorePacketAssemble` |
| `LocalWrite` | staged 数据已足够覆盖当前目标 `line/subtile/meta packet` | 执行 1 次本地写动作 | 对应 `remaining_*writes` 成功减 1 | 返回 `FetchFromTMEM` 或进入 `Completed` |
| `LocalRead` | 仅 `StoreC` 使用，且 staged store packet 队列为空 | 从 `CMem` 读取 1 个 subtile | 成功读到 1 个 subtile | `StorePacketAssemble` |
| `StorePacketAssemble` | 已读到 1 个或 2 个 subtile | 把 subtile 重新组织成一个或多个 staged store packet | staged store packet 队列非空 | `WriteBackToTMEM` |
| `WriteBackToTMEM` | staged store packet 队列非空 | 向 `TMEM` 写 1 个 packet | 成功写完 1 个 packet | 返回 `LocalRead` 或进入 `Completed` |
| `Completed` | 所有 `remaining_*` 已归零，且 staged store packet 队列为空 | 更新 slot 聚合位，推进 `pending_mem_ops_` | 出队 | - |

#### 8.6.2 `FillA` 的精度相关子状态

##### `A fp8 dense`

1. 初始:
   - `remaining_tmem_reads = 4`
   - `remaining_amem_fill_lines = 4`
   - `remaining_metamem_fill_packets = 0`
2. 每读 1 个 packet，staged buffer 增加 1 个 payload packet。
3. 只要 staged packet 足够覆盖当前目标 line，就可执行 1 次 `AMem line write`。
4. 最短序列:
   - `read p0 -> write line0 -> read p1 -> write line1 -> read p2 -> write line2 -> read p3 -> write line3`

##### `A fp16 dense`

1. 初始:
   - `remaining_tmem_reads = 8`
   - `remaining_amem_fill_lines = 4`
2. 每条 `AMem line` 需要 2 个 packet。
3. 最短序列:
   - `read p0 -> read p1 -> write line0`
   - `read p2 -> read p3 -> write line1`
   - `read p4 -> read p5 -> write line2`
   - `read p6 -> read p7 -> write line3`

##### `A sparse`

1. 在对应 dense 路径基础上，还要再经历:
   - `read meta packet`
   - `MetaMem write`
2. 当前 meta 的传输不是“每 16B 一次”，而是 1 个完整 `64B packet`。

#### 8.6.3 `FillA` 的分阶段进入/退出条件

| 阶段 | 进入条件 | 退出条件 |
| --- | --- | --- |
| `Fetch payload packet` | `remaining_tmem_reads > remaining_metamem_fill_packets`，且 staged payload 不足以支撑当前 `AMem line` | 成功读到 1 个 payload packet，或本拍 `TMEM read egress` / bank 不可用 |
| `Write AMem line` | staged payload packet 已满足当前目标 `line` 的需要，且 `AMem` 本地写端口可用 | 成功写完 1 条 `line`，`remaining_amem_fill_lines--` |
| `Fetch meta packet` | payload line 全部写完，且 `remaining_metamem_fill_packets != 0` | 成功读到 1 个 meta packet |
| `Write MetaMem` | meta packet 已 staged，且 `MetaMem` 本地写端口可用 | 成功写完 1 个 meta packet，`remaining_metamem_fill_packets--` |

#### 8.6.4 `FillB` 的精度相关子状态

`FillB` 与 `FillA dense` 类似，只是最终落地对象变成 `BMem line0..line3`。

#### 8.6.5 `FillC` 的精度相关子状态

##### `C fp8`

1. 初始:
   - `remaining_tmem_reads = 4`
   - `remaining_cmem_fill_subtiles = 4`
2. 两个 packet 共同覆盖一组 16 列数据，可依次写两个 subtile。
3. 最短序列:
   - `read p0 -> read p1 -> write subtile0 -> write subtile1`
   - `read p2 -> read p3 -> write subtile2 -> write subtile3`

##### `C fp16`

1. 初始:
   - `remaining_tmem_reads = 8`
   - `remaining_cmem_fill_subtiles = 4`
2. 每组 4 个 packet 覆盖两个 subtile。

##### `C fp32`

1. 初始:
   - `remaining_tmem_reads = 16`
   - `remaining_cmem_fill_subtiles = 4`
2. 每组 8 个 packet 覆盖两个 subtile。

#### 8.6.6 `FillC` 的分阶段进入/退出条件

| 阶段 | 进入条件 | 退出条件 |
| --- | --- | --- |
| `Fetch payload packet` | `remaining_tmem_reads != 0`，且当前 staged payload 不足以支撑当前目标 `subtile` | 成功读到 1 个 payload packet，或本拍 `TMEM read egress` / bank 不可用 |
| `Write C subtile` | staged payload 已满足当前目标 `subtile`，且 `CMem` 本地写端口可用 | 成功写完 1 个 `subtile`，`remaining_cmem_fill_subtiles--` |

#### 8.6.7 `StoreC` 状态

1. 初始:
   - `remaining_cmem_dump_subtiles = 4`
   - `remaining_tmem_writes = packet_count(fmt_d)`
2. 每读取两个 subtile，会生成一组 staged store packet。
3. staged packet 未 drain 完前，不继续读取后续 subtile。

#### 8.6.8 `StoreC` 的分阶段进入/退出条件

| 阶段 | 进入条件 | 退出条件 |
| --- | --- | --- |
| `Read C subtile` | `remaining_cmem_dump_subtiles != 0`，且 staged store packet 队列为空 | 成功读到 1 个 `subtile`，或本拍 `CMem read` 端口不可用 |
| `Assemble store packet` | 已读到足够的 `subtile` 形成一组输出 packet | staged store packet 队列非空 |
| `Write TMEM packet` | staged store packet 队列非空，且 `TMEM write ingress` 与相关 bank 端口可用 | 成功写完 1 个 packet，`remaining_tmem_writes--` |

#### 8.6.9 `MemUop` complete 的判定

一个 `MemUop` 只有在以下条件同时满足时才 complete:

1. `remaining_tmem_reads == 0`
2. `remaining_tmem_writes == 0`
3. `remaining_amem_fill_lines == 0`
4. `remaining_bmem_fill_lines == 0`
5. `remaining_cmem_fill_subtiles == 0`
6. `remaining_cmem_dump_subtiles == 0`
7. `remaining_metamem_fill_packets == 0`
8. staged store packet 队列为空
9. 不再持有左半组 `StoreC` 暂存状态

当前实现对应 `mem_op_complete()`。

### 8.7 `WMMA job` 状态机

#### 8.7.1 入队

当 `WMMA` 在 `execute()` 中通过合法性检查后:

1. 创建一个 async op
2. `pending_wmma_uops_[async_id] = 8`
3. 把一个 `PendingWmmaJob` 压入 `pending_wmma_jobs_`
4. `a_slot.wmma_pending = true`
5. `b_slot.wmma_pending = true`
6. `c_slot.c_wmma_inflight++`
7. `c_slot.cmem_final_valid = false`

#### 8.7.2 从 `pending_wmma_jobs_` 进入 `active_wmma_job_`

当前转移条件非常明确:

1. 当前没有 active job
2. `pending_wmma_jobs_` 非空
3. `tensorcore_.ready(true)` 为真

只有三者同时满足，队首 job 才会被提升为 `active_wmma_job_`。

补充说明:

1. 当前实现同时最多只有一个 `active_wmma_job_`
2. 当前实现不会在 active job 尚未完成 8 条 primitive 发射时切换到下一个 job
3. 因此 `pending_wmma_jobs_` 是宏级 FIFO，`active_wmma_job_` 是唯一拥有 primitive issue 权的上下文

#### 8.7.3 active job 发射 primitive

进入 active 后:

1. 每拍最多发 1 条 primitive
2. 发射顺序固定为 `uop0..uop7`
3. 当 `job.next_uop == 8` 时，A/B slot 的 `wmma_pending` 清零，并结束 active job

#### 8.7.4 active job 的 stall 条件

即使当前 `pending_wmma_jobs_` 非空，只要以下任一条件不满足，本拍都不会发新的 primitive:

1. `tensorcore_.ready(true)` 为真
2. 若当前没有 active job，则队首 pending job 能在本拍被提升为 active

当前实现中不会在 `issue_wmma_primitives()` 里对 A/B line 做二次逐 line ready 检查；它依赖更早的 `a_ready/b_ready/c_ready` 聚合位保证数据已到位。这也是当前“slot 级控制 + line/subtile 级存储 valid”混合模型的一部分。

#### 8.7.5 primitive retire 与宏完成

1. primitive 在 `advance_tensorcore_pipeline()` 尾部 retire
2. 每 retire 一条，`pending_wmma_uops_[async_id]--`
3. 只有最后一条 primitive retire 后:
   - `pending_wmma_uops_[async_id]` 归零
   - `c_wmma_inflight--`
   - `c_dirty=true`
   - `cmem_final_valid=true`
   - `async_tensor_complete(async_id)`

#### 8.7.6 8 条 primitive 与 4 条目标 line / 4 个 subtile 的固定映射

当前 `WMMA` 的 8 条 primitive 固定等于 `2 个 k-slice x 4 个 C subtile`。对应关系如下:

| primitive | `A line` | `B line` | `C subtile` |
| --- | --- | --- | --- |
| `uop0` | `A line0` | `B line0` | `subtile0` |
| `uop1` | `A line0` | `B line2` | `subtile1` |
| `uop2` | `A line2` | `B line0` | `subtile2` |
| `uop3` | `A line2` | `B line2` | `subtile3` |
| `uop4` | `A line1` | `B line1` | `subtile0` |
| `uop5` | `A line1` | `B line3` | `subtile1` |
| `uop6` | `A line3` | `B line1` | `subtile2` |
| `uop7` | `A line3` | `B line3` | `subtile3` |

从硬件视角看:

1. `A line0..line3` 表示 `(m,k)` 二维分块
2. `B line0..line3` 表示 `(n,k)` 二维分块
3. `C subtile0..subtile3` 表示 `(m,n)` 二维分块
4. 每个 `C subtile` 都要累加两个 `k-slice`，因此总共是 8 条 primitive

### 8.8 AsyncTensorOp 状态机

`AsyncTensorOp` 是 `Core` 侧 async 主线事务描述符，适用于 `TMA_LOAD / TMA_STORE / TMEM_SHIFT / MmaLoad / MmaStore / Wmma` 的 waiter、barrier 和完成通知跟踪。

#### 8.8.1 通用状态

| 状态 | 硬件语义 | 进入条件 | 退出条件 |
| --- | --- | --- | --- |
| `Issued` | 指令当拍刚建立事务，尚未开始 async 推进 | `Core::execute()` 接受指令 | 到达 `first_service_cycle` |
| `LaunchDelay` | 事务已可被检查，但仍在消耗启动延迟 | `remaining_launch_cycles > 0` | `remaining_launch_cycles == 0` |
| `Transfer` | packet 正在 ingress/egress/shift 路径上逐拍前进 | 启动延迟结束 | 相关 `remaining_tmem_*` 归零 |
| `Completed` | 事务自身已完成 | 所有必要 packet 传输与后处理结束 | 等待 waiter / barrier 消费 |
| `Finalized` | waiter 与 barrier 已收到完成事件 | `finalize_async_tensor_op()` 执行 | 从活跃表移除 |

#### 8.8.2 `first_service_cycle` 的 contract

1. `TMA_LOAD/TMA_STORE/TMEM_SHIFT` 在 issue 当拍只建立 `AsyncTensorOp`，`first_service_cycle = issue_cycle + 1`
2. 因此它们最早在下一全局拍的 `advance_async_tensor_engine()` 中才开始真正推进
3. `MmaLoad/MmaStore/Wmma` 的 `first_service_cycle` 设为极大值，它们的 complete 事件不由 `advance_async_tensor_engine()` 主动推进，而是由 `TensorUnit` 在对应事务完成时调用 `async_tensor_complete()`

## 9. 逐指令时序契约

以下契约统一按八个问题回答:

1. 操作对象是什么
2. 何时允许 issue
3. issue 当拍改哪些状态
4. 下一拍进入什么阶段
5. in-flight 如何逐拍推进
6. 何时 complete
7. 何时 visible / consumable
8. 当前 Cmodel 如何实现

### 9.1 TMEM_ALLOC

#### 操作对象

1. `TMEM` allocation table

#### 允许 issue

1. allocator 未被 `TMEM_REL_PERMIT` seal
2. `col_span != 0`
3. `col_span` 不超过 TMEM 可分配容量

#### issue 当拍

1. 直接分配一个 allocation
2. 立即生成 handle
3. 至少创建 `valid / col_span / payload_col_base`

#### 下一拍与 in-flight

1. `TMEM_ALLOC` 不进入 async 主线
2. 不存在 packet 传输阶段

#### complete

1. issue 即 complete

#### visible / consumable

1. allocation table 中的新条目对同拍后续 `execute()` 路径可见
2. 这是“内部资源状态可见”，不是“寄存器依赖已经解除”
3. handle 作为寄存器结果能否被后续指令安全消费，仍取决于普通流水的 `commit/scoreboard release`

#### 实现映射

1. [execute.cpp](/mnt/d/wode_code_trunk/vortex/sim/simx/execute.cpp) 调用 `core_->tmem_alloc(...)`
2. [core.cpp](/mnt/d/wode_code_trunk/vortex/sim/simx/core.cpp) 中 `tmem_alloc()` 直接调用 `tmem_.alloc(...)`

### 9.2 TMEM_FREE

#### 操作对象

1. `TMEM` allocation table

#### 允许 issue

1. handle 有效
2. handle 当前未处于不可释放状态

#### issue/complete/visible

1. issue 即释放 allocation
2. 同拍之后新分配可复用该空间

### 9.3 TMEM_REL_PERMIT

#### 操作对象

1. allocator seal 状态

#### issue/complete/visible

1. issue 即 complete
2. 同拍起新的 `TMEM_ALLOC` 不再允许成功

### 9.4 TMA_LOAD

#### 操作对象

1. 一个 `TMEM` handle
2. 一个 `window_id`
3. 一个 `TmaDescriptor`

#### 允许 issue

1. descriptor 可读
2. handle 有效
3. 目标 window 已绑定或可按 descriptor 绑定

#### issue 当拍

1. 创建 `AsyncTensorOp(TmaLoad)`
2. 将 `payload_ready=false`
3. 若有 metadata，则 `meta_ready=false`
4. 记录 `window_id / desc_id / issue_cycle / first_service_cycle`

#### 下一拍进入什么阶段

1. issue 当拍只建立 async op
2. 最早下一全局拍，在 `advance_async_tensor_engine()` 中进入 `LaunchDelay` 或直接进入 packet 传输

#### in-flight 推进

1. 若 `remaining_launch_cycles > 0`，先逐拍消耗启动延迟
2. 启动延迟结束后，按 payload packet 逐拍通过 TMEM write ingress
3. 若有 metadata，再按 meta packet 逐拍通过 TMEM write ingress
4. 每个 packet 还必须通过相关 bank 的 `1W` 预算

#### complete

1. payload packet 全部写入目标 window
2. metadata packet 全部写入 meta shadow window
3. 若有附加流程，也必须全部结束

#### visible / consumable

1. complete 当拍重新置 `payload_ready=true`
2. 若有 metadata，当拍重新置 `meta_ready=true`
3. complete 当拍 `layout_epoch++`
4. 这些变化对同拍后续 `execute()/issue()/schedule()` 可见

#### 实现映射

1. issue: [core.cpp](/mnt/d/wode_code_trunk/vortex/sim/simx/core.cpp) 的 `tma_load()`
2. 推进: `advance_one_async_tensor_transaction()`
3. 启动延迟字段: `remaining_launch_cycles`

### 9.5 TMA_STORE

#### 操作对象

1. 一个 `TMEM` handle/window
2. 一个 `TmaDescriptor`

#### 允许 issue

1. descriptor 可读
2. handle/window 有效
3. 源 window 当前可读

#### issue 当拍

1. 创建 `AsyncTensorOp(TmaStore)`
2. 当前实现会把对应 `payload_ready` 或 `meta_ready` 拉低，表示该 window 正处于导出过程中

#### 下一拍与 in-flight

1. 先消耗 `remaining_launch_cycles`
2. 再按 shaped window 的 packet 语义逐 packet 读取 `TMEM`
3. 每拍最多通过 `1 read egress packet`

#### complete

1. 所有 packet 均已读出并写回外部内存

#### visible

1. complete 当拍进入 waiter / barrier 完成判定

### 9.6 TMEM_SHIFT

#### 操作对象

1. 一个 `TMEM` handle
2. 指定 `window_id`
3. 可选 top-row refill descriptor

#### 对外语义

对外冻结语义为:

1. 针对指定 `window_id`
2. 按**数学矩阵视图**
3. 将所有数学行整体下移 1 行
4. 顶部数学行可选择清零或由 refill descriptor 回填

这不是“logical line 下移 1 行”的对外 contract。

#### 允许 issue

1. handle/window 有效
2. 源 window 当前可读

#### issue 当拍

1. 创建 `AsyncTensorOp(TmemShift)`
2. `payload_ready=false`
3. 若有 refill，记录 `refill_descriptor_id`

#### 下一拍与 in-flight

1. 若无 refill，进入纯 shift packet 阶段
2. 若有 refill，先进入 `LaunchDelay`，然后再进入 top-row 回填 packet 阶段
3. shift 本体需要占用 `TMEM` 读写资源
4. refill top row 仍通过 `TMEM write ingress`

#### complete

1. 全部数学行 shift 完成
2. 若有 refill，顶部数学行所有回填 packet 都已写完

#### visible / consumable

1. complete 当拍 `payload_ready=true`
2. complete 当拍 `layout_epoch++`
3. 同拍后续 `execute()/issue()/schedule()` 可见

### 9.7 MMA_LOAD

#### 操作对象

1. 一个 `TMEM handle`
2. 一个 `window_id`
3. 一个数学 `tile_id`
4. 一组 `A slot / B slot / C slot` 中的某一个

#### 对外控制信息

当前显式携带:

1. `target`
2. `window_id`
3. `tile_id`
4. `slot_id`

`packet_id` 不对外暴露，只在内部 `MemUop` 中推进。

#### 允许 issue

1. handle 有效
2. 对应 window 有效
3. `payload_ready/meta_ready` 满足要求
4. 对应 target 的 slot 可绑定或可复用

细化条件:

1. `MMA_LOAD(A dense)` 要求 `payload_ready`
2. `MMA_LOAD(A sparse)` 要求 `payload_ready && meta_ready`
3. `MMA_LOAD(B)` 要求 `payload_ready`
4. `MMA_LOAD(C)` 要求 `payload_ready`

#### issue 当拍

1. 对应 target 建立一个 `MemUop::FillA/FillB/FillC`
2. 对应 slot 的 `*_pending=true`
3. 若需要重绑定 slot，则先清理该 slot 对应的本地存储窗口

#### 下一拍与 in-flight

1. 因为 `TensorUnit::tick()` 在平台级先于 `Core::tick()`，所以 issue 当拍并不会立刻被 `advance_tensor_memory_pipeline()` 处理。
2. 最早下一全局拍，该 `MemUop` 才会出现在 `advance_tensor_memory_pipeline()` 中。
3. 之后按以下顺序推进:
   - `TMEM packet fetch`
   - packet 进入 staged buffer
   - 满足条件后执行本地 line/subtile/packet 写入

#### complete

1. 当前 target 的所有 packet 都已从 `TMEM` 取到
2. 当前 target 的所有本地写入动作都已完成
3. 若 `target=A sparse`，还必须包括 meta packet 写入完成

#### visible / consumable

1. `MMA_LOAD(A/B/C)` complete 之后，对应 `a_ready/b_ready/c_ready` 置位
2. 只有聚合 ready 置位后，该 slot 才能被 `WMMA` 或 `MMA_STORE` 消费

#### 实现映射

1. issue: [execute.cpp](/mnt/d/wode_code_trunk/vortex/sim/simx/execute.cpp) -> [tensor_unit.cpp](/mnt/d/wode_code_trunk/vortex/sim/simx/tensor_unit.cpp) `enqueue_async_mma_load()`
2. 状态推进: `advance_tensor_memory_pipeline()`

### 9.8 WMMA

#### 操作对象

1. 一个 `A slot`
2. 一个 `B slot`
3. 一个 `C slot`
4. 一个宏级 `WMMA job`

#### 允许 issue

1. `A/B/C slot` 已 valid
2. `a_ready && b_ready && c_ready`
3. `a_slot.wmma_pending == false`
4. `b_slot.wmma_pending == false`
5. `c_slot.store_pending == false`

#### issue 当拍

1. 创建一个宏级 `WMMA job`
2. 将其压入 `pending_wmma_jobs_`
3. `pending_wmma_uops_[async_id] = 8`
4. `a_slot.wmma_pending = true`
5. `b_slot.wmma_pending = true`
6. `c_slot.c_wmma_inflight++`
7. `c_slot.cmem_final_valid = false`
8. `c_slot.c_dirty = true`

#### 下一拍与 in-flight

1. 因为 `TensorUnit::tick()` 先于 issue 当拍的 `Core::tick()`，新 job 最早下一全局拍才会被 `issue_wmma_primitives()` 看到。
2. 之后若 `active_wmma_job_` 为空且 `tensorcore_.ready(true)`，队首 pending job 会变成 active job。
3. active job 每拍最多发 1 条 primitive。
4. 发射顺序固定为 `uop0..uop7`。

#### retire

1. primitive 只在 `advance_tensorcore_pipeline()` 尾部 retire
2. retire 时直接累加到 `CMem(fp22)`，不经过独立 accumulator

#### complete

只有以下条件全部满足，宏 `WMMA` 才 complete:

1. 8 条 primitive 全部发射
2. 8 条 primitive 全部 retire
3. `pending_wmma_uops_[async_id] == 0`

#### visible / consumable

1. 最后一个 primitive retire 当拍:
   - `c_wmma_inflight--`
   - `c_dirty=true`
   - `cmem_final_valid=true`
   - `async_tensor_complete(async_id)`
2. 但这些变化发生在 `TensorUnit::tick()` 尾部，因此要到下一全局拍才会影响新的 `MMA_STORE issue`

### 9.9 MMA_STORE

#### 操作对象

1. 一个 `C slot`
2. 一个 `tile_id`
3. 一个目标 `window`

#### 允许 issue

1. `target=C`
2. `c_slot.cmem_final_valid == true`
3. `c_slot.c_wmma_inflight == 0`
4. `c_slot.store_pending == false`

#### issue 当拍

1. enqueue 一个 `MemUop::StoreC`
2. `store_pending = true`

#### 下一拍与 in-flight

1. issue 当拍之后，最早下一全局拍才开始 `CMem subtile read`
2. 读到一个 subtile 后，立即生成该 subtile 对应的 staged store packet
3. staged packet 逐个写回目标 window

#### complete

1. 当前 tile 的所有 store packet 已完成写回
2. `store_pending=false`
3. 若该 slot 已无其他 dirty 数据，则后续可复用

#### visible / consumable

1. `MMA_STORE` complete 之后，TMEM 中该 tile 的 packet 才允许被后续 `TMA_STORE` 或其他消费者读取
2. 同一拍的 `MMA_STORE complete` 对更早执行的本拍 `advance_tensor_memory_pipeline()` 不可见

### 9.10 TC_COMMIT

#### 语义

1. 把当前 warp 尚未 completed、尚未 committed 的 async tensor op 绑定到指定 barrier
2. `pending_tx += committed_count`

#### complete / visible

1. issue 即 complete
2. barrier 的 `pending_tx` 同拍可见

### 9.11 TC_FENCE

#### BEFORE

1. 当前实现中 `TC_FENCE.BEFORE` 立即返回成功
2. 原因是 in-order warp issue 已经保证 fence 前后的相对顺序

#### AFTER

1. 若当前 warpgroup 没有 pending committed async op，则立即成功
2. 否则把整个 warpgroup 标记为 fence waiter
3. 当 `try_resume_fence_waiters()` 发现条件满足时再统一恢复

### 9.12 TC_WAIT

当前 `TC_WAIT` 只等待本 warpgroup 上以下 async op:

1. `MmaLoad`
2. `MmaStore`
3. `Wmma`

它**不等待** `TMA_LOAD / TMA_STORE / TMEM_SHIFT`。这些操作应通过 `TMA_WAIT` 或 `TC_COMMIT + mbarrier` 等待。

### 9.13 TMA_WAIT

1. `TMA_WAIT(async_id)` 等待指定 async tensor op 完成
2. 若该 async op 已完成，则立即成功
3. 若未完成，则当前 warpgroup 挂到 `async_tensor_waiters_[async_id]`
4. 完成后由 `resume_async_waiters(async_id)` 恢复

### 9.14 MBAR_INIT / MBAR_ARRIVE / MBAR_WAIT

barrier phase 完成条件冻结为:

1. `pending_arrivals == 0`
2. `pending_tx == 0`

`MBAR_WAIT` 语义:

1. 若当前 phase 已完成，则立即通过
2. 否则当前 warpgroup 挂到 `waiters_bitmap`
3. phase 完成后通过 `try_complete_mbarrier()` 统一恢复

## 10. 最短完成周期表

以下“最短完成周期”全部指**无资源冲突、无 bank 冲突、队列中没有更早事务、descriptor/window 已就绪**时的理论下界。

### 10.1 TMA 类

| 指令 | 公式 | 说明 |
| --- | --- | --- |
| `TMA_LOAD` | `L + P + M` | `L=launch latency`, `P=payload packets`, `M=meta packets` |
| `TMA_STORE` | `L + P` | `L=launch latency`, `P=payload packets` |
| `TMEM_SHIFT(no refill)` | `S` | `S=shift window packet count` |
| `TMEM_SHIFT(with refill)` | `L + S + R` | `L=refill launch latency`, `R=refill packets` |

### 10.2 MMA_LOAD

| target | 格式 | packet 数 | 本地目标 | 最短完成周期 |
| --- | --- | --- | --- | --- |
| `A/B` | `fp8` | 4 | 4 lines | 8 |
| `A/B` | `fp16` | 8 | 4 lines | 12 |
| `A sparse` | `fp8` | 4 + 1 meta | 4 lines + 1 meta packet | 10 |
| `A sparse` | `fp16` | 8 + 1 meta | 4 lines + 1 meta packet | 14 |
| `C` | `fp8` | 4 | 4 subtiles | 8 |
| `C` | `fp16` | 8 | 4 subtiles | 12 |
| `C` | `fp32` | 16 | 4 subtiles | 20 |

### 10.3 WMMA

设:

1. `U = 8`，即 primitive 数
2. `D_tc` 为 TensorCore 从 `push_uop()` 到首个 primitive retire 的固定流水深度

则:

1. 宏 `WMMA` 最短 issue-to-last-issue 为 `8` 拍
2. 宏 `WMMA` 最短 issue-to-complete 为 `U + D_tc - 1`

说明:

1. `D_tc` 不由本文重新定义，仍以当前 `open_tensorcore` 流水实现为准。
2. 本文冻结的是“存在一个固定计算流水深度”，以及“8 条 primitive 全部 retire 后宏 `WMMA` 才 complete”。

### 10.4 MMA_STORE

| 格式 | `CMem` 读取对象 | TMEM 写 packet 数 | 最短完成周期 |
| --- | --- | --- | --- |
| `fp8` | 4 subtiles | 4 | 8 |
| `fp16` | 4 subtiles | 8 | 12 |
| `fp32` | 4 subtiles | 16 | 20 |

## 11. 同拍可见性矩阵

| 生产事件 | 发生位置 | 同拍对谁可见 | 说明 |
| --- | --- | --- | --- |
| `TMEM_ALLOC complete` | `Core::execute()` | 同拍后续内部资源查询 | 仅 allocation table 可见，寄存器依赖未必已解除 |
| `TMA_LOAD complete` | `advance_async_tensor_engine()` | 同拍 `execute()/issue()/schedule()` | `payload_ready/meta_ready/layout_epoch` 同拍更新 |
| `TMA_STORE complete` | `advance_async_tensor_engine()` | 同拍 wait/barrier 恢复 | 结果写回已经完成 |
| `TMEM_SHIFT complete` | `advance_async_tensor_engine()` | 同拍 `execute()/issue()/schedule()` | `payload_ready/layout_epoch` 同拍更新 |
| `MMA_LOAD complete` | `TensorUnit::tick()` 早期 | 同拍稍后的 `Core::tick()` | 因为平台级 `TensorUnit` 先于 `Core` |
| primitive retire | `advance_tensorcore_pipeline()` | 同拍稍后的 `Core::tick()` 可见；下一拍本地 store 可见 | 对同拍更早的 `advance_tensor_memory_pipeline()` 不可见 |
| `WMMA complete` | `advance_tensorcore_pipeline()` 尾部 | 下一拍 `MMA_STORE issue` | 因为 `MMA_STORE` 新 issue 要等下一次 `Core::execute()` |
| `TMA_WAIT/TC_WAIT` 恢复 | `finalize_async_tensor_op()` | 同拍 `schedule()` | waiters 恢复早于 barrier `pending_tx--` |

## 12. 同拍恢复顺序

当前 `finalize_async_tensor_op()` 顺序冻结为:

1. `resume_async_waiters(async_id)`
2. `try_resume_fence_waiters()`
3. 若该 op 已 `committed` 且 barrier 有效，则 `pending_tx--`
4. `try_complete_mbarrier(barrier_id)`

这意味着:

1. async waiters 的恢复早于 barrier `pending_tx--`
2. fence 恢复也早于 barrier 完成判定
3. barrier 完成恢复发生在更靠后的步骤

waiter 唤醒顺序按 `wid` 递增扫描。

## 13. Scoreboard、busy 位与等待规则

### 13.1 Scoreboard 不是 tensor 完成条件

寄存器 scoreboard 只表示寄存器冲突是否解除，不代表 tensor 事务是否真正完成。

以下状态不能由 scoreboard 替代:

1. `payload_ready / meta_ready`
2. `cmem_final_valid`
3. `c_wmma_inflight`
4. async op `completed`
5. barrier phase `completed`

### 13.2 Handle busy 与 slot busy 必须分离

必须区分两类 busy:

1. `handle busy`
   由 `TMA_LOAD / TMA_STORE / TMEM_SHIFT` 占用 handle/window 引起
2. `slot busy`
   由本地 `a_pending/b_pending/c_pending/wmma_pending/c_wmma_inflight/store_pending` 引起

这两类冲突在 scheduler 和 execute 中都必须分开判断，不能合并成单一 busy 位。

## 14. 面向新读者的总时序摘要

若只关心主路径 `TMA_LOAD -> MMA_LOAD -> WMMA -> MMA_STORE -> TMA_STORE`，当前总时序可概括为:

1. `TMA_LOAD` 在 `Core::execute()` 当拍 issue，但真正的 packet 推进要等下一全局拍 `advance_async_tensor_engine()`。
2. `MMA_LOAD` 在 `Core::execute()` 当拍 issue，但真正的本地搬运要等下一全局拍 `TensorUnit::advance_tensor_memory_pipeline()`。
3. `MMA_LOAD` complete 后，聚合 `a_ready/b_ready/c_ready` 才会置位。
4. `WMMA` 只在聚合 ready 全部满足时，才会把宏指令变成内部 8 条 primitive。
5. primitive 在 TensorCore 中逐拍发射，逐条 retire，并直接累加进 `CMem(fp22)`。
6. `MMA_STORE` 只有在 `cmem_final_valid=true` 且 `c_wmma_inflight==0` 后，才能开始导出。
7. `TMA_STORE` 与 `TMEM_SHIFT` 始终走 `Core` 侧 async 主线，而不走 `TensorUnit` 本地队列。

### 14.1 例 1: `A fp16 dense` 从 `MMA_LOAD` 到 `WMMA`

以下例子只说明最短无冲突下界，不考虑更早队列项、bank 冲突和 scheduler 抢占。

假设:

1. 某个 `A slot` 当前空闲
2. `MMA_LOAD(target=A, fmt=fp16)` 在全局 cycle `N` issue
3. 该 tile 的 `A` 数据已在 `TMEM` 中 ready

则最短路径如下:

1. `cycle N`
   - `Core::execute()` issue `MMA_LOAD(A)`
   - 建立一个 `FillA` `MemUop`
   - `a_pending=true`
2. `cycle N+1`
   - 先执行 `TensorUnit::tick()`
   - `advance_tensor_memory_pipeline()` 看到该 `FillA`
   - 读入 `p0`
3. `cycle N+2`
   - 读入 `p1`
4. `cycle N+3`
   - 写 `A line0`
5. `cycle N+4 ~ N+11`
   - 以同样方式完成 `line1/line2/line3`
6. `cycle N+12`
   - `FillA complete`
   - `a_pending=false`
   - `a_ready=true`
   - 因为 `TensorUnit::tick()` 先于当拍 `Core::tick()`，所以同拍稍后的 `Core::execute()/schedule()` 已能看到 `a_ready=true`
7. 若 `B slot` 与 `C slot` 也已 ready，则同拍可以通过 `Core::execute()` issue 一个新的宏 `WMMA`
8. 该 `WMMA` 最早在 `cycle N+13` 被 `TensorUnit::issue_wmma_primitives()` 接住，进入 primitive 发射

这个例子说明两个关键点:

1. `MMA_LOAD` 与 `WMMA` 可以在同一全局拍“前者 complete、后者 issue”
2. 但 `WMMA` 的 primitive 真正开始发射一定是在再下一全局拍，因为平台级顺序总是 `TensorUnit::tick()` 早于 `Core::tick()`

### 14.2 例 2: `TMEM_SHIFT + refill top row`

假设:

1. 某个 shaped window 已经 ready
2. 在全局 cycle `M` issue `TMEM_SHIFT(window_id, refill_desc)`
3. shift 本体需要 `S` 个 packet，refill top row 需要 `R` 个 packet，refill TMA 启动延迟为 `L`

则最短路径如下:

1. `cycle M`
   - `Core::execute()` issue `TMEM_SHIFT`
   - 建立 `AsyncTensorOp`
   - `payload_ready=false`
2. `cycle M+1`
   - `advance_async_tensor_engine()` 开始处理该 op
   - 若 `remaining_launch_cycles != 0`，先进入 refill 的 `LaunchDelay`
3. 后续若存在 `L` 拍启动延迟，则这 `L` 拍只消耗 `remaining_launch_cycles`
4. shift 本体阶段:
   - 每拍消耗 1 组 `TMEM read + write` 资源
   - 共持续 `S` 个 packet
5. refill 阶段:
   - 每拍通过 `TMEM write ingress` 写 1 个 top-row packet
   - 共持续 `R` 个 packet
6. 最终:
   - `payload_ready=true`
   - `layout_epoch++`
   - 同拍后续 `execute()/issue()/schedule()` 可见

这个例子说明:

1. `TMEM_SHIFT` 的外部语义是“数学行下移 1 行”，不是“logical line 下移 1 行”
2. 但真正的硬件实现仍然是由 planner/resolver 把数学行解释成具体 packet 和逻辑地址，再逐 packet 完成 shift 与 refill

## 15. 必须记录的 trace 事件

后续 RTL / Cmodel 对齐时，至少比较以下事件:

1. `TMEM_ALLOC(handle, col_span)` issue / return
2. `TMA_LOAD(async_id, handle, window_id, desc_id)` issue / complete
3. `TMA_STORE(async_id, handle, window_id, desc_id)` issue / complete
4. `TMEM_SHIFT(async_id, handle, window_id, refill_descriptor_id)` issue / complete
5. `MMA_LOAD(handle, window_id, tile_id, slot_id, target)` issue / local-complete
6. `WMMA(async_id, a_slot, b_slot, c_slot)` issue / first-primitive-issue / last-primitive-retire / complete
7. `MMA_STORE(handle, window_id, tile_id, slot_id)` issue / complete
8. `payload_ready / meta_ready / layout_epoch` 更新
9. `a_ready / b_ready / c_ready / cmem_final_valid / c_wmma_inflight` 更新
10. `mbarrier.pending_arrivals / pending_tx / phase` 更新

## 16. 本版本不再签核的内容

以下内容即使代码中仍保留兼容入口，也不属于当前 timing contract:

1. 旧 non-descriptor tensor memory 指令
2. tagged-handle `window_id`
3. 未绑定 shaped window 的 legacy linear packet 语义
4. 尚未重新对齐的 sparse compute datapath

## 17. 签核使用方法

使用本文对拍时，推荐顺序如下:

1. 先比全局平台顺序、`Core::tick()` 与 `TensorUnit::tick()` 顺序是否一致
2. 再比 `TMA_LOAD / TMEM_SHIFT / MMA_LOAD / WMMA / MMA_STORE` 的 issue 与 complete 事件
3. 再比 `ready / valid / dirty / inflight / epoch` 等关键状态位
4. 最后才比总 cycles 和 IPC

如果总周期不一致，但上述事件顺序、状态转移、资源预算与可见性边界一致，则优先认为问题出在资源参数或仲裁实现细节，而不是时序契约本身。
