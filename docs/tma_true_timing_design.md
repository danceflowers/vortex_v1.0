# TMA True-Timing Data Visibility Design

本文记录本轮对 SIMX TMA global memory 访问时序建模的实现分析和修改结果。

## 目标

原来的 `cp.async.bulk.tensor` CModel 路径更接近功能模型：指令发起后立即把 global memory payload 拷贝到 LMEM。这样旧测试能够通过，但不能表达真实 TMA 时序，因为后续 `tmem_cp` / MMA 可能在 global memory 数据尚未返回时就看到 LMEM payload。

本轮目标是让 TMA 的 descriptor 和 payload 都通过 SimPort 进入 socket-to-L2 路径，并且只在 L2 timing response 返回后才让数据对后续指令可见。

核心语义：

- `cp.async.bulk.tensor` 发起时只创建 TMA pending operation。
- Tensor descriptor 不在 issue 阶段同步解析，而是先通过 L2 timing request 读取 descriptor。
- Descriptor timing response 全部返回后，TMA 才解析 `tensor_map_t` 并展开 payload request。
- Payload timing response 返回后，TMA 才执行功能性 `dcache_read -> lmem_write`。
- Payload 全部完成后，TMA 才向 Core 发 async completion。
- Core 收到 completion 后才触发 `mbarrier.complete_tx`，从而唤醒等待数据可见性的 warp。

## 为什么接到 L2 而不是 L1

TMA 在硬件语义上是独立数据搬运单元，不应伪装成普通 core load/store 进入 L1 D-cache pipeline。否则会把 TMA 流量和 core scalar/vector load/store 混在一起，容易错误表达：

- TMA bypass L1 的行为。
- TMA 和 L1 miss/writeback 在 socket-to-L2 入口处竞争带宽。
- TMA completion 由 TMA 自己的 async engine 管理，而不是由普通 load/store pipeline 管理。

因此本轮采用的连接方式是：

- TMA request 从 `Tma::CacheReqOut` 发出。
- 在 socket-to-L2 port 前新增一个小仲裁器 `tma_l2_arb`。
- 仲裁器 input 0 接原来的 L1 miss/writeback 路径。
- 仲裁器其他 input 接各 core 的 TMA request port。
- 仲裁器 output 接原来的 `Socket::mem_req_ports[i]`，也就是进入 cluster/L2 的入口。

这等价于“绕过 L1，进入 socket 到 L2 的入口仲裁器”。代码中原来没有这个显式入口仲裁器，所以本轮新增了它。

相关代码：

- `sim/simx/socket.cpp`
- `sim/simx/core.h`
- `sim/simx/tensor/tma.h`
- `sim/simx/tensor/tma.cpp`

## TMA 内部状态机

本轮在 `Tma` 中新增了两层 pending state。

### `pending_cpabulk_transfer_t`

表示一条完整的 `cp.async.bulk.tensor` transfer。

关键字段：

- `async_id`：Core 分配的异步操作 ID。
- `is_store`：区分 global->LMEM load 和 LMEM->global store。
- `tensor_map_addr`：global memory 中 descriptor 地址。
- `coords[5]`：从 LMEM args 中读取的坐标参数。
- `global_addr`：descriptor 返回后计算出的 payload global 地址。
- `lmem_addr`：payload 的 LMEM 目标/源地址。
- `total_bytes`：descriptor 返回后计算出的 payload 总字节数。
- `descriptor_next_offset` / `descriptor_completed_bytes`：descriptor request 进度。
- `payload_next_offset` / `payload_completed_bytes`：payload request 进度。
- `inflight_cache_requests`：当前 transfer 还未返回的 L2 timing request 数量。
- `tx_bound_mbar`：如果指令使用 `complete_tx`，记录要更新的 mbarrier 地址。

### `pending_cpabulk_cache_request_t`

表示一个 cache-line 粒度的 L2 timing request。

关键字段：

- `async_id`：归属的 TMA transfer。
- `stage`：`Descriptor` 或 `Payload`。
- `offset`：当前 request 在 descriptor/payload 中的偏移。
- `bytes`：当前 request 覆盖的字节数。

## 发起 `cp.async.bulk.tensor`

入口在 `Tma::cpabulk_tensor_load()` / `Tma::cpabulk_tensor_store()`。

现在 issue 阶段只做这些事：

1. 检查 args 指针必须指向 LMEM。
2. 从 LMEM 读取 `cpabulk_transfer_args_t`。
3. 记录 `smem_addr`、coords、mbarrier 地址等本地参数。
4. 创建 `pending_cpabulk_transfer_t`。

注意：issue 阶段不再同步读取 global memory 中的 `tensor_map_t`，也不再同步搬运 payload。

原因是 descriptor 本身位于 global memory，如果要表达真实时序，它也必须等 L2 response 返回后才可被 TMA 解析。

## Descriptor 阶段

`Tma::advance_cpabulk_transfer_ops()` 每个 tick 推进 pending transfer。

如果 transfer 的 `descriptor_ready == false`：

1. 按 cache line 边界切分 descriptor request。
2. 通过 `CacheReqOut[0]` 发 `MemReq`。
3. request 进入 socket-to-L2 入口仲裁。
4. L2 response 回到 `CacheRspIn[0]`。

`Tma::drain_cache_responses()` 收到 descriptor response 后：

1. 累加 `descriptor_completed_bytes`。
2. 当 descriptor 128B 全部返回后，才执行功能性 `dcache_read(&tmap, tensor_map_addr, sizeof(tmap))`。
3. 解析 `tensor_map_t`，计算：
   - `total_bytes`
   - payload `global_addr`
4. 设置 `descriptor_ready = true`。

这样 payload 阶段不会早于 descriptor timing completion。

## Payload 阶段和数据可见性

当 `descriptor_ready == true` 后，TMA 开始发 payload request。

payload 同样按 cache line 边界切分。每个 request 返回后才执行实际数据搬运。

对于 TMA load：

```cpp
core_->dcache_read(buf, op.global_addr + request.offset, request.bytes);
core_->lmem_write(buf, op.lmem_addr + request.offset, request.bytes);
```

对于 TMA store：

```cpp
core_->lmem_read(buf, op.lmem_addr + request.offset, request.bytes);
core_->dcache_write(buf, op.global_addr + request.offset, request.bytes);
```

Store payload request 现在是真正的 write request：

```cpp
mem_req.write = true;
mem_req.write_response = true;
```

`write_response` 是本轮新增的 per-request completion 标志。普通 L1 writeback/write-through 请求默认不设置它，因此不会被迫接收 write ack；TMA store 设置它，让 cache/L2 在该写请求达到模型定义的可见性点后返回一个不带数据的 write ack。TMA 收到这个 ack 后才执行功能性 `LMEM -> global` 写入，并推进 payload 完成计数。

关键点：payload response 返回前，LMEM 不会被写入。因此后续 `tmem_cp` 或 MMA 不能依赖“旧模型的立即可见性”。

对于 store，对应的关键点是：write ack 返回前，TMA store 不会被视为完成，也不会向 Core 发 async completion。

## Write Ack 建模

本轮没有打开 `CacheSim::Config::write_reponse` 这个全局开关。原因是普通 cache writeback 也会收到 response，而上游 L1 cache 的 fill/MSHR 路径会把这类 response 当成 read fill，语义会混乱。

实际实现采用 per-request 标志：

- `MemReq::write_response`：请求方是否需要 write completion ack。
- `MemRsp::write`：response 是否是 write ack。

Cache 侧行为：

- read request 仍然正常返回 read response。
- write request 默认不返回 response。
- 如果 `config.write_reponse` 或 `MemReq::write_response` 为真，则返回 write ack。
- write ack 保留原请求的 `tag/cid/uuid`，TMA 用 `tag` 匹配 pending cache request。

MemSim 侧行为：

- read request 一直返回 response。
- write request 只有在 `write_response=true` 时才返回 response。
- `ProcessorImpl` 的内存 pending-read 计数只在 read response 上递减，避免 write ack 误伤 read pending 计数。

## Completion 和 mbarrier

TMA payload 全部完成后，通过 `AsyncOpCompletionOut` 通知 Core。

本轮扩展了 `TensorAsyncOpCompletion`，让 TMA completion 能携带：

- `tx_bound_mbar`
- `tx_bytes`
- `payload_size_bytes`

Core 在 `drain_tma_completion_notices()` 中把这些信息写回 `AsyncTensorOp`，然后调用 `async_tensor_complete()`。

`async_tensor_complete()` 再调用 `on_async_tensor_op_completed()`，其中如果 `tx_bound_mbar != 0 && tx_bytes != 0`，就执行：

```cpp
mbarrier_complete_tx(op.tx_bound_mbar, op.tx_bytes);
```

因此 `mbarrier.wait` 只有在真实 TMA payload 完成后才能通过。

## 测试同步方式调整

旧测试在 `cpabulk_tensor_ld()` 后直接使用 LMEM payload。真实时序下这是错误的，因为 LMEM 数据尚未保证可见。

本轮将相关 kernel 改为：

1. `mbarrier_init(mbar_addr, 1)`
2. `mbarrier_expect_tx(mbar_addr, expected_payload_bytes)`
3. 使用 `cpabulk_tensor_ld_complete_tx(...)` 发 TMA load
4. `mbarrier_arrive_token(mbar_addr)`
5. `mbarrier_wait(mbar_addr, phase)`
6. wait 通过后才执行 `tmem_cp` / MMA

涉及测试：

- `tests/regression/tcgen05_mma_minimal/kernel.cpp`
- `tests/regression/tcgen05_mma_extended/kernel.cpp`

## 修复 extended sparse case 的 tx 字节数

调试 `tcgen05_mma_extended` 时，case 2 最初出现 hang。

watchdog dump 显示：

- 4 个 TMA async op 都已经 completed。
- TMA 内部没有 pending request。
- mbarrier 状态为 `expected_tx=2112`，但 `pending_tx=1600`。

差值是 512B。

根因是 sparse case 的 C payload 格式是 FP16，host 实际生成 512B；kernel 中原来按 dense FP32 C payload 的 1024B 计算 expected tx，所以 mbarrier 永远等不到最后 512B。

修复方式：

```cpp
static inline uint32_t c_payload_bytes(uint32_t case_id) {
  return is_sparse_case(case_id) ? TCGEN05_PAYLOAD_BYTES : TCGEN05_C_PAYLOAD_BYTES;
}
```

然后用：

```cpp
uint32_t tma_tx_bytes =
    (2 * TCGEN05_PAYLOAD_BYTES) + c_payload_bytes(arg->case_id);
```

sparse case 再额外加 `TCGEN05_META_BYTES`。

## 当前实现的限制

当前实现已经满足 descriptor/payload 的 L2 timing 可见性，但仍有一些边界限制：

- TMA store 现在使用 per-request write ack；普通 cache writeback 默认仍然不产生 ack。
- TMA request 的粒度按 `MEM_BLOCK_SIZE` 切分，没有建模更复杂的 sector/sub-partition 行为。
- Descriptor 解析仍通过功能性 `dcache_read` 完成，但读取时机已经被 descriptor timing response gate 住。
- 当前只接了 `CacheReqOut[0]` / `CacheRspIn[0]`，没有把 TMA 多端口化。

这些限制不影响当前 regression 的 true-timing 数据可见性语义。

## 验证结果

本轮验证命令：

```bash
make -C runtime/simx VORTEX_HOME=$PWD -j2
env VORTEX_PROFILING=3 make -C tests/regression/tcgen05_mma_minimal VORTEX_HOME=$PWD run-simx
env VORTEX_PROFILING=3 make -C tests/regression/tcgen05_mma_extended VORTEX_HOME=$PWD run-simx
env VORTEX_PROFILING=3 make -C tests/regression/tcu_mbarrier_pending_tx VORTEX_HOME=$PWD run-simx
env VORTEX_PROFILING=3 make -C tests/regression/tcu_ldst VORTEX_HOME=$PWD run-simx
env VORTEX_PROFILING=3 make -C tests/regression/tcu_tma_store_ack VORTEX_HOME=$PWD run-simx
git diff --check -- kernel/include/vx_tensor.h sim/simx/common/types.h sim/simx/cache_sim.cpp sim/simx/core.cpp sim/simx/core.h sim/simx/mem_sim.cpp sim/simx/processor.cpp sim/simx/socket.cpp sim/simx/tensor/tma.cpp sim/simx/tensor/tma.h sim/simx/tensor/tensor_mem_port_types.h tests/regression/tcgen05_mma_extended/kernel.cpp tests/regression/tcgen05_mma_minimal/kernel.cpp tests/regression/tcu_tma_store_ack
```

验证结果：

- `runtime/simx` 编译通过。
- `tcgen05_mma_minimal` 通过。
- `tcgen05_mma_extended` 三个 case 全部通过。
- `tcu_mbarrier_pending_tx` 通过。
- `tcu_ldst` 通过。
- `tcu_tma_store_ack` 通过，覆盖 TMA store write ack completion 路径。
- `git diff --check` 通过。

## 涉及源码文件

主要源码修改：

- `kernel/include/vx_tensor.h`
- `sim/simx/common/types.h`
- `sim/simx/cache_sim.cpp`
- `sim/simx/core.cpp`
- `sim/simx/core.h`
- `sim/simx/mem_sim.cpp`
- `sim/simx/processor.cpp`
- `sim/simx/socket.cpp`
- `sim/simx/tensor/tma.cpp`
- `sim/simx/tensor/tma.h`
- `sim/simx/tensor/tensor_mem_port_types.h`
- `tests/regression/tcgen05_mma_extended/kernel.cpp`
- `tests/regression/tcgen05_mma_minimal/kernel.cpp`
- `tests/regression/tcu_tma_store_ack/`

其中核心实现集中在 `sim/simx/tensor/tma.cpp` 和 `sim/simx/socket.cpp`。
