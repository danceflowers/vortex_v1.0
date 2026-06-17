# SIMX Tensor Naming Rules

版本: v1.0

日期: 2026-03-31

适用范围: [sim/simx](/mnt/d/wode_code_trunk/vortex/sim/simx) 中 `TMEM / TMA / TensorUnit / OpenTensorCore local SRAM / decode / execute / timing contract` 相关代码与文档。

本文冻结 tensor 路径的命名与注释规则，目标是：

1. 名字先对应硬件对象，再对应软件实现。
2. 相同术语在代码、文档、trace 中保持同义。
3. 避免把“目标对象”和“时序动作”混为一谈。

## 1. 总原则

1. 名字优先表达硬件语义，不优先表达代码实现细节。
2. 一个名字只描述一层粒度，不跨层混用。
3. 目标对象、总线传输、时钟拍次必须分开命名。
4. 模糊词只允许出现在局部上下文非常明确的地方，默认禁止。

## 2. 固定术语

### 2.1 三层视图

1. `math` / `matrix`
   表示数学矩阵元素视图。
2. `logical`
   表示 TMEM 逻辑视图中的 `(column, line)`。
3. `physical`
   表示 TMEM 物理 bank / row / byte 视图。

禁止：

1. 用 `bank` 指代逻辑 `column`
2. 用 `row` 同时指代数学矩阵行和物理 SRAM 行

### 2.2 数据粒度

1. `window`
   表示一个 TMEM 逻辑窗口。
2. `tile`
   表示数学矩阵语义下的 `16x16` 基本块。
3. `packet`
   表示 `64B` 总线传输单位。
4. `line`
   只用于 `AMem/BMem` 目标行，或 TMEM 逻辑视图中的逻辑行。
5. `subtile`
   只用于 `CMem` 的 `8x8` C 子块。
6. `primitive`
   表示 `WMMA` 拆出的单条 TensorCore 基本计算单元工作项。

禁止：

1. 用 `beat` 指代 `AMem/BMem` 的目标行
2. 用 `beat` 指代 `CMem` 的 `subtile`
3. 用 `row` 替代 `subtile`

### 2.3 时序粒度

1. `cycle`
   只表示全局时钟拍。
2. `launch latency`
   表示命令发出后、packet 传输开始前的固定前置延迟。
3. `packet transfer`
   表示一次 `64B` packet 级前进。
4. `local port action`
   表示一次本地 SRAM 读或写动作。

`beat` 只允许在以下场景出现：

1. 明确表示某条总线的一次 beat 传输
2. 明确表示某个接口协议里的 beat 概念

## 3. 状态词规则

### 3.1 允许使用的状态词

1. `pending`
   已 admit，但还没 complete。
2. `ready`
   已满足下游启动条件。
3. `inflight`
   已进入执行/传输过程，尚未 complete。
4. `completed`
   内部状态机已完成。
5. `valid`
   某份数据当前存在且可被读出。
6. `dirty`
   数据已被修改，尚未导出或覆盖。
7. `final`
   对外可作为最终结果消费。

### 3.2 推荐组合

1. `payload_ready`
2. `meta_ready`
3. `cmem_final_valid`
4. `store_pending`
5. `remaining_tmem_read_packets`
6. `remaining_amem_fill_lines`
7. `remaining_cmem_fill_subtiles`

禁止：

1. `done_flag`
2. `busy_flag`
3. `tmp_ready`
4. `state_ok`

## 4. 函数命名规则

### 4.1 推进函数

推进某一级硬件流水或后台引擎时，优先使用：

1. `advance_*`
2. `issue_*`
3. `retire_*`
4. `enqueue_*`
5. `mark_*`

例如：

1. `advance_async_tensor_engine`
2. `advance_one_async_tensor_transaction`
3. `encode_math_window_packet`
4. `resolve_window_linear_packet_region`
5. `shift_window_math_rows_down`
6. `advance_tensor_memory_pipeline`
7. `advance_tensorcore_pipeline`
8. `issue_wmma_primitives`

不建议继续使用：

1. `service_*`
2. `tick_*`
3. `dispatch_*`  
   除非语义确实是硬件 dispatch 仲裁

### 4.2 目标对象函数

1. `AMem/BMem`
   使用 `line`
   例如：
   - `fill_lines()`
   - `packets_per_fill_line()`
   - `write_fill_line()`
2. `CMem`
   使用 `subtile`
   例如：
   - `fill_subtiles()`
   - `dump_subtiles()`
   - `write_fill_subtile()`
3. `MetaMem`
   使用 `packet`
   例如：
   - `fill_packets()`
   - `write_fill_packet()`

## 5. 计数器命名规则

所有 `remaining_*` 都必须满足：

1. 明确对象
2. 明确动作
3. 明确粒度

推荐形式：

1. `remaining_<module>_<action>_<unit>`

例如：

1. `remaining_tmem_read_packets`
2. `remaining_tmem_write_packets`
3. `remaining_amem_fill_lines`
4. `remaining_bmem_fill_lines`
5. `remaining_cmem_fill_subtiles`
6. `remaining_cmem_dump_subtiles`
7. `remaining_metamem_fill_packets`
8. `remaining_launch_cycles`

禁止：

1. `remaining_reads`
2. `remaining_writes`
3. `remaining_base_cycles`
4. `remaining_data`

## 6. 控制字与 descriptor 命名规则

### 6.1 控制字

控制字局部变量必须带上来源和用途，不允许统一叫 `control`。

推荐：

1. `tmem_control_word`
2. `mma_memory_control_word`
3. `wmma_slot_control_word`

### 6.2 字段 helper

提取 bitfield 时统一使用：

1. `*_field_mask`
2. `*_ctl_*`
3. `*_make_control`

例如：

1. `tcu_control_field_mask`
2. `tcu_tmem_op_ctl_window_id`
3. `tcu_mma_mem_ctl_tile_id`

若 ABI 历史字段无法立即更名，必须在注释中明确说明其当前语义。当前典型例子是：

1. `bank_span`
   字段名保留，但当前语义应明确写为“logical TMEM column span”，不能再解释成 physical bank span

## 7. 模块边界端口命名规则

### 7.1 端口类型头文件

若一个头文件只定义跨模块端口事务类型，而不实现真正的总线、仲裁器或链路行为，文件名必须使用：

1. `*_port_types.h`
2. 或 `*_protocol.h`

不建议继续使用：

1. `*_bus.h`

因为 `bus` 更适合表示真正的共享互连或总线实现，而不是单纯的端口事务类型定义。

当前 tensor 路径中的典型例子是：

1. `tensor_mem_port_types.h`

### 7.2 端口事务类型

跨模块事务类型统一使用：

1. `*Req`
2. `*Rsp`
3. `*Completion`

例如：

1. `TensorMemPortReq`
2. `TensorMemPortRsp`
3. `TensorAsyncOpCompletion`

字段命名必须说明语义，不使用模糊的 `kind/data/payload/control` 兜底：

1. `access_type`
2. `arbitration_age`
3. `port_request`
4. `write_packet`
5. `read_packet`

### 7.3 端口实例名

端口实例统一用方向表达：

1. `*ReqOut`
2. `*ReqIn`
3. `*RspOut`
4. `*RspIn`
5. `*CompletionOut`
6. `*CompletionIn`

例如：

1. `TensorMemReqOut`
2. `TensorMemRspIn`
3. `TensorAsyncOpCompletionOut`
4. `tensor_mem_req_in_`
5. `tensor_mem_rsp_out_`
6. `tensor_async_op_completion_in_`

## 8. slot 命名规则

1. `slot`
   默认指 operand slot。
2. 需要区分时，必须明确写成：
   - `A slot`
   - `B slot`
   - `C slot`
3. 若是数组或状态组，优先写：
   - `a_slots_`
   - `b_slots_`
   - `c_slots_`

禁止：

1. “对应 slot”而不说明是 A/B/C 哪一类
2. 用一个 `slot` 同时指 A/B/C 三类对象

## 9. 注释规则

### 8.1 必须加注释的地方

1. 文件头  
   说明该文件对应哪个硬件模块。
2. 状态机结构体  
   说明对象生命周期和职责。
3. 关键推进函数  
   说明该函数对应哪一段硬件推进。
4. 控制字拆解处  
   说明该控制字选择的对象和粒度。
5. 固定映射关系  
   例如 `WMMA` 的 `8 primitive <-> 4 A lines / 4 B lines / 4 C subtiles`

### 8.2 不建议加的注释

1. 逐行翻译代码语句
2. 重复变量名本身已经表达的意思
3. 用中文或英文写含义模糊的“这里做一下处理”

## 10. 推荐示例

### 10.1 推荐

```cpp
// Advance all Core-side asynchronous tensor transactions by one cycle.
void Core::advance_async_tensor_engine();

// Advance one already-issued Core-side async tensor transaction by one cycle.
void Core::advance_one_async_tensor_transaction(...);

// Write one AMem destination line for the current FillA transaction.
void AMem::write_fill_line(...);

// Resolve one window-linear packet into a logical (column, line) region.
bool Tmem::resolve_window_linear_packet_region(...);

uint32_t remaining_tmem_read_packets = 0;
uint32_t next_payload_packet_idx = 0;
struct TensorMemPortReq;
```

### 10.2 不推荐

```cpp
void service_mem_ops();
uint32_t remaining_reads = 0;
uint32_t control = rs2_data.at(thread_start).u32;
uint32_t fill_beats() const;
struct TensorMemBus;
```

## 11. 文档同步规则

代码命名更新后，以下文档必须同步：

1. [tensor_timing_contract.md](/mnt/d/wode_code_trunk/vortex/docs/tensor_timing_contract.md)
2. 设计报告中与当前实现绑定的章节
3. 相关 regression 注释或 helper 说明

若代码与文档命名冲突，以代码为准，但必须在同一改动中补齐文档，不允许长期漂移。
