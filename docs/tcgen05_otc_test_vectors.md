# tcgen05 / OpenTensorCore Test Vectors and Coverage

本文档记录两类测试：

- OTC standalone：直接驱动 OpenTensorCore TensorCoreTop Cmodel，验证 16x16 宏级 MMA 被拆成 8x8 primitive 后的计算行为。
- Vortex+TC kernel：通过 Vortex SIMX 执行 RISC-V kernel，验证 tcgen05 兼容指令在 core、TensorUnit、TMEM、LMEM、mbarrier 和 DRAM 路径上的端到端行为。

验证日期：2026-05-07  
默认舍入：RNE  
OTC standalone 固定随机种子：SEED=1  
Vortex SIMX warp/thread 配置：NUM_THREADS=32

## 1. Reference 构建方式

### 1.1 OTC standalone 的 fp32 golden reference

OTC standalone 的 golden reference 使用 fp32 矩阵乘构建，但输入值不是直接使用原始随机 double，而是先经过测试目标格式量化，再按 OTC 入口数据格式解释：

1. A、B 的原始测试值先分别量化为测试指定格式，例如 FP8_E4M3 或 FP16。
2. 进入 TensorCoreTop 前，A、B 会转换为 OTC 计算入口使用的 FP9 表示。golden reference 使用 FP9 反解出的数值作为有效输入值。
3. 对每个输出元素执行 fp32 标量累加：

   ```text
   acc_fp32 = C_effective
   for k in K:
       acc_fp32 += float(A_effective[m,k]) * float(B_effective[k,n])
   golden[m,n] = acc_fp32
   ```

4. C 的有效值来自测试指定 C 格式转换后的值。在 OTC primitive 接口上，C 操作数以 final-add 输入形式参与本次 primitive 计算；golden 先按相同格式得到 C 的有效数值，再用 fp32 参与累加。
5. 实际输出来自 TensorCoreTop retire 的 FP22 结果，并按 D 精度转换为 FP16 或 FP32 后再与 golden 比较。

这意味着 golden 对齐的是“格式量化后的输入 + fp32 数学矩阵乘”，而不是原始 double 随机数矩阵。这样可以避免把输入格式量化误差误判为 OTC 计算误差。

### 1.2 Vortex+TC kernel 的 golden reference

Vortex+TC kernel 测试分两类：

- 数值 MMA 测试使用 host 侧 fp32 reference。host 生成 A/B/C 的可解释矩阵，按 kernel 输入格式打包到 DRAM payload，同时构建两组 reference：
  - golden1_quantized_input：A/B/C 先按指令目标输入格式量化，再把量化后的有效数值转回 fp32 做 A*B+C。这组 reference 对齐硬件实际消费的 FP8/FP16/FP32 payload。
  - golden2_raw_fp32_input：A/B/C 使用 host 生成的原始 fp32 数值直接做 A*B+C，不经过 FP8/FP16 量化回读。这组 reference 用来观察目标数学表达式本身和当前测试向量量化后的差异。
- 指令/同步/搬运测试使用 bit-exact reference。比如 TCU_ST/TCU_LD round-trip 直接要求每个 lane 读回自己的 32-bit pattern；mbarrier pending_tx 测试要求 LMEM 目标区首个 u32 等于 DRAM 源数据首元素。

## 2. OTC Standalone 测试

测试入口：

```bash
make -C sim/simx/tensor/open_tensorcore run TEST_ID=0 SEED=1
```

测试实现位置：

- sim/simx/tensor/open_tensorcore/tensor_helper/test/main.cpp
- sim/simx/tensor/open_tensorcore/Makefile

### 2.1 Case 1: Dense MMA 基础测试

测试名称：dense_mma_fp8_fp8_to_fp16

测试目标：

验证 OTC Cmodel 的基础 dense MMA + C 输入累加路径。该测试确认 16x16x16 宏级矩阵乘可以拆成 4 个 8x8 输出 subtile 和 2 个 K-slice primitive，并正确执行 A*B+C。

测试输入：

- A：16x16，FP8_E4M3，identity-like 矩阵。A[m,m]=1，其余元素为 0。
- B：16x16，FP8_E4M3，确定性随机非负矩阵，数值范围约为 [0, 0.25]，先量化为 FP8_E4M3。
- C：16x16，FP16，确定性随机非负矩阵，数值范围约为 [0, 0.25]。

测试配置：

- 测试对象：OTC Cmodel / TensorCoreTop
- 矩阵规模：M=16，N=16，K=16
- 数据精度：FP8_E4M3 * FP8_E4M3 -> FP16
- sparse 模式：dense
- K tile：单个 16-wide K tile
- 阈值：max_abs <= 0.01

测试指令序列：

1. Host harness 生成 A/B/C test vector。
2. A/B 按输入精度量化，再转换为 FP9 primitive 输入。
3. 16x16 宏 MMA 被拆成 4 个 8x8 输出 subtile。
4. 每个 subtile 先把 C 转为 FP22 partial，再对 K=0..15 分两次 8-wide K-slice 调用 TensorCoreTop push_uop。
5. TensorCoreTop tick 到 retire，输出 FP22。
6. FP22 输出按 D=FP16 转换后与 fp32 golden 比较。

期望输出：

A 是 identity-like 矩阵，因此 D 应等于 B 的对应 16x16 FP8 量化有效值再加 C，经过 FP16 输出格式表示后与 fp32 golden 的误差小于阈值。

实际结果：

```text
cycles=104
max_abs=0.000122
mean_abs=0.000036
rms=0.000060
PASS
```

### 2.2 Case 2: 非对称精度 MMA 测试

测试名称：asym_mma_fp16_fp8_to_fp32

测试目标：

验证 OTC Cmodel 能接收非对称输入精度并执行 C 输入累加：A 为 FP16，B 为 FP8_E4M3，C/D 为 FP32。该测试主要覆盖 A/B 不同输入格式转换后进入 FP9 primitive 的路径、C 的 FP32 输入路径，以及 FP32 输出转换路径。

测试输入：

- A：16x16，FP16，identity-like 矩阵。A[m,m]=1，其余元素为 0。
- B：16x16，FP8_E4M3，确定性随机非负矩阵，数值范围约为 [0, 0.25]。
- C：16x16，FP32，确定性随机非负矩阵，数值范围约为 [0, 0.25]。

测试配置：

- 测试对象：OTC Cmodel / TensorCoreTop
- 矩阵规模：M=16，N=16，K=16
- 数据精度：FP16 * FP8_E4M3 -> FP32
- sparse 模式：dense
- 阈值：max_abs <= 0.01

测试指令序列：

1. A 先量化为 FP16，B 先量化为 FP8_E4M3。
2. A/B 分别转换为 TensorCoreTop 入口 FP9。
3. 16x16 宏 MMA 拆成 8x8 primitive。
4. 每个 subtile 先把 C 转为 FP22 partial，再让 primitive 通过 push_uop 进入 TensorCoreTop。
5. TensorCoreTop retire FP22，harness 将 FP22 转换为 FP32 输出。
6. 输出矩阵与 fp32 golden 比较。

期望输出：

由于 A 是 identity-like 矩阵，D 应等于 B 的有效值矩阵加 C，并以 FP32 输出。误差应小于阈值。

实际结果：

```text
cycles=104
max_abs=0.000023
mean_abs=0.000006
rms=0.000009
PASS
```

### 2.3 Case 3: 多 K-tile 累加测试

测试名称：multi_ktile_accum_fp8_fp8_to_fp16

测试目标：

验证 K 维分块累加。该测试构造两个 active K tile，使输出必须经过多次 primitive 累加，目标表达式为：

```text
D = A0 * B0 + A1 * B1 + C
```

测试输入：

- A：16x64，FP8_E4M3。A 在 K=0..15 和 K=32..47 两段分别放置 identity-like 子矩阵，其余 K 段为 0。
- B：64x16，FP8_E4M3，确定性随机非负矩阵，范围约为 [0, 0.25]。
- C：16x16，FP16，确定性随机非负矩阵，范围约为 [0, 0.25]。

测试配置：

- 测试对象：OTC Cmodel / TensorCoreTop
- 矩阵规模：M=16，N=16，K=64
- 数据精度：FP8_E4M3 * FP8_E4M3 -> FP16
- sparse 模式：dense
- macro K tile：32
- 阈值：max_abs <= 0.03

测试指令序列：

1. 初始化 partial = C。
2. 对 K=0..63 按 8-wide K-slice 拆分。
3. 对每个输出 subtile，依次发射多个 TensorCoreTop primitive。
4. 每次 primitive 的 FP22 retire 结果作为下一次 primitive 的 C 输入，实现原位式 partial 累加。
5. 最终 FP22 partial 转换为 FP16 输出。
6. 与 fp32 golden 的 A0*B0 + A1*B1 + C 比较。

期望输出：

只有两个 K tile 对结果有贡献，因此每个输出元素应等于对应两段 B 的和再加 C。输出转换为 FP16 后与 fp32 golden 的误差小于阈值。

实际结果：

```text
cycles=416
max_abs=0.000244
mean_abs=0.000062
rms=0.000085
PASS
```

### 2.4 Case 4: 2:4 sparse MMA 测试

测试名称：sparse_2_4_mma_fp8_fp8_to_fp16

测试目标：

验证 2:4 structured sparsity metadata 解析与 sparse routing 功能。该测试确认压缩 A payload 和 metadata 可以恢复出正确的 dense A 位置，并选择 B 中对应 K 行参与计算。

测试输入：

- A：16x32，FP8_E4M3。每 4 个连续 K 元素中恰好 2 个非零，其余为 0。
- B：32x16，FP8_E4M3，确定性随机非负矩阵，范围约为 [0, 0.25]。
- C：16x16，FP16，确定性随机非负矩阵，范围约为 [0, 0.25]。
- Metadata：每行每个 4-wide group 编码两个非零 lane index；每个 8x8 primitive 使用 16B metadata line。

测试配置：

- 测试对象：OTC Cmodel / TensorCoreTop + sparse_select routing
- 矩阵规模：M=16，N=16，K=32
- 数据精度：FP8_E4M3 * FP8_E4M3 -> FP16
- sparse 模式：2:4
- 阈值：max_abs <= 0.05

测试指令序列：

1. Host harness 生成满足 2:4 约束的 sparse A。
2. A 被压缩为每个 4-wide group 保留 2 个 payload 元素。
3. Metadata 记录每个 group 中两个非零 lane 的 index。
4. sparse routing 根据 metadata 将 compact A 还原到 8-wide K primitive 拓扑，并选择 B 中对应 K 行。
5. routed A/B 进入 TensorCoreTop dense primitive 计算。
6. 多个 K-slice 的结果通过 FP22 partial 累加，并加上 C。
7. 最终 FP22 转 FP16 输出，与 fp32 golden 比较。

期望输出：

```text
D = sparse(A) * B + C
```

golden 使用 sparse(A) 的实际非零位置和 B 的有效量化值进行 fp32 矩阵乘。

实际结果：

```text
cycles=208
max_abs=0.000244
mean_abs=0.000066
rms=0.000080
PASS
```

### 2.5 Case 5: 1:4 sparse MMA 测试

测试名称：sparse_1_4_mma_fp8_fp8_to_fp16

测试目标：

验证 1:4 metadata 与 K 扩展。该测试覆盖更长 K=64 的 sparse 累加，同时每 4 个连续 K 元素只保留 1 个非零 A 元素。

测试输入：

- A：16x64，FP8_E4M3。每 4 个连续 K 元素中恰好 1 个非零，其余为 0。
- B：64x16，FP8_E4M3，确定性随机非负矩阵，范围约为 [0, 0.25]。
- C：16x16，FP16，确定性随机非负矩阵，范围约为 [0, 0.25]。
- Metadata：每行每个 4-wide group 编码一个非零 lane index；每个 8x8 primitive 使用 16B metadata line。

测试配置：

- 测试对象：OTC Cmodel / TensorCoreTop + sparse_select routing
- 矩阵规模：M=16，N=16，K=64
- 数据精度：FP8_E4M3 * FP8_E4M3 -> FP16
- sparse 模式：1:4
- macro K tile：32
- 阈值：max_abs <= 0.05

测试指令序列：

1. Host harness 生成满足 1:4 约束的 sparse A。
2. A 被压缩为每个 4-wide group 保留 1 个 payload 元素。
3. Metadata 记录每个 group 中非零 lane 的 index。
4. sparse routing 根据 metadata 将 compact A 还原到 8-wide K primitive 拓扑，并选择 B 中对应 K 行。
5. routed A/B 进入 TensorCoreTop dense primitive。
6. K=64 被拆为多个 8-wide K-slice，partial 在 FP22 中连续累加。
7. 最终 FP22 转 FP16 输出，与 fp32 golden 比较。

期望输出：

```text
D = sparse(A) * B + C
```

golden 使用 sparse(A) 的实际非零位置和 B 的有效量化值进行 fp32 矩阵乘。

实际结果：

```text
cycles=416
max_abs=0.000320
mean_abs=0.000065
rms=0.000082
PASS
```

## 3. Vortex+TC Kernel 测试

测试入口：

```bash
make -C tests/regression/tcgen05_mma_minimal run-simx
make -C tests/regression/tcgen05_mma_extended run-simx
make -C tests/regression/tcu_ldst run-simx
make -C tests/regression/tcu_mbarrier_pending_tx run-simx
```

### 3.1 tcgen05_mma_minimal

测试目标：

验证 tcgen05 风格 descriptor MMA 的端到端 kernel 路径。该测试覆盖 DRAM payload、tensor_map 描述符、cp.async.bulk.tensor、LMEM、tcgen05.cp、TMEM、TCU_MMA、mbarrier 和 TCU_LD 回读。

测试输入：

- A：16x16，FP16 identity 矩阵。
- B：16x16，FP16 矩阵，host 生成公式为 `(((k * 3 + n) & 0x7) + 1) / 100.0`。
- A/B payload：各 512B，按 tcgen05 16x16 fp16 tile layout 打包。
- 输出缓冲区：16x16 FP32，初值为 -1.0。

测试配置：

- 测试对象：Vortex SIMX + TensorUnit + OpenTensorCore
- kernel：tests/regression/tcgen05_mma_minimal
- grid：1x1x1
- block：32x1x1
- 矩阵规模：M=16，N=16，K=16
- idesc：FP16 * FP16，C/D 为 FP32，M=16，N=16
- tensor_map：A/B 各一个 128B rank-1 tensor_map，element_type=F16，box_size=256，global_stride=2B，element_stride=1
- cpabulk_transfer_args：A/B 各一个 32B LMEM 参数块，保存 LMEM payload 目标地址，coords 全 0
- TMEM allocation：A col_span=16，D col_span=32
- 阈值：max_abs <= 0.01

测试指令序列：

1. cp.async.bulk.tensor 加载 A payload：DRAM -> LMEM。
2. cp.async.bulk.tensor 加载 B payload：DRAM -> LMEM。
3. TMEM_ALLOC 分配 A 和 D 的 TMEM 地址。
4. tcgen05.cp shape=4 将 A 从 LMEM shared descriptor 指向的地址写入 TMEM。
5. 在 LMEM 构造 operand_block，携带 A 的 TMEM 地址与 B 的 shared descriptor。
6. 构造 idesc，指定 FP16 * FP16 -> FP32。
7. mbarrier_init 初始化同步对象。
8. TCU_MMA no_accum 发射 MMA。
9. mbar_commit 提交异步 tensor op。
10. mbarrier_arrive_token / mbarrier_wait 等待 MMA 完成。
11. mbar_fence_after 保证后续读结果可见。
12. 每个 lane 通过 8 次 TCU_LD 从 TMEM 读回 D 的 8 个 u32 word，并写入 DRAM 输出。
13. TMEM_DEALLOC 释放 D/A。

期望输出：

由于 A 是 identity，D 应等于 B 的 FP32 表示。host 使用 fp32 矩阵乘构建 golden：

```text
ref[m,n] = sum_k A_fp32[m,k] * B_fp32[k,n]
```

实际结果：

```text
instrs=34483
cycles=36314
IPC=0.949579
max_abs_error=0.001875
mean_abs_error=0.000839843
rms_error=0.000990524
PASS
```

### 3.2 tcgen05_mma_extended

测试目标：

验证 tcgen05.mma 的扩展 kernel 路径。该测试在真实 Vortex SIMX kernel 中覆盖三类能力：

- FP16*FP8->FP32 非对称输入精度，确认 idesc.kind=F16F8 和 operand_block fmt_cd 能被 tcdecode 解析并送入 TensorUnit。
- 2:4 sparse MMA，确认 TCU_MMA.sp、idesc.sparsity_kind=0、A compact payload、metadata packet 和 B dense payload 能完成端到端稀疏 routing。
- 1:4 sparse MMA，确认 TCU_MMA.sp、idesc.sparsity_kind=1 能选择 1:4 metadata 解释方式。

测试输入：

- Case 1 A：16x16 FP16 identity 矩阵。
- Case 1 B：16x16 FP8_E4M3 确定性矩阵，host 生成公式为 `(((k * 5 + n * 3) & 0x7) + 1) / 16.0`。
- Case 1 C：16x16 FP32 确定性矩阵，host 生成公式为 `(((m * 7 + n * 3) & 0x7) + 1) / 64.0`。
- Case 2 A：16x16 FP8_E4M3 2:4 structured sparse 矩阵。每个 4-wide K group 恰好 2 个非零元素。
- Case 3 A：16x16 FP8_E4M3 1:4 structured sparse 矩阵。每个 4-wide K group 恰好 1 个非零元素。
- Case 2/3 B：16x16 FP8_E4M3 确定性矩阵，host 生成公式为 `(((k * 3 + n * 5) & 0x7) + 1) / 32.0`。
- Case 2/3 C：16x16 FP16 确定性矩阵，host 生成公式为 `(((m * 5 + n * 3) & 0x7) + 1) / 64.0`。
- Sparse metadata：64B metadata payload，逻辑上划分为 4 条 16B metadata line，对应 2 个 M subtile x 2 个 K phase。

测试配置：

- 测试对象：Vortex SIMX + tcdecode + TensorUnit + OpenTensorCore
- kernel：tests/regression/tcgen05_mma_extended
- grid：1x1x1
- block：32x1x1
- 矩阵规模：M=16，N=16，K=16
- Case 1 idesc：F16F8，A=FP16，B=FP8，C/D=FP32，dense
- Case 2 idesc：F8F6F4，A=FP8，B=FP8，C/D=FP16，TCU_MMA.sp，sparsity_kind=2:4
- Case 3 idesc：F8F6F4，A=FP8，B=FP8，C/D=FP16，TCU_MMA.sp，sparsity_kind=1:4
- A/B/C tensor_map：rank-1 tensor_map，按 payload byte size 搬运到 LMEM
- metadata tensor_map：rank-1 64B U8 tensor_map，只在 sparse case 使用
- TMEM allocation：A col_span=16，D col_span=32
- C 输入 TMEM 位置：C payload 先写入 D TMEM；TCU_MMA 通过 enable-input-d 从同一个 D taddr 读取 C，再把最终 D 写回该位置
- sparse metadata TMEM 位置：A payload 后的 64B metadata packet

测试指令序列：

1. cp.async.bulk.tensor 加载 A payload：DRAM -> LMEM。
2. cp.async.bulk.tensor 加载 B payload：DRAM -> LMEM。
3. cp.async.bulk.tensor 加载 C payload：DRAM -> LMEM。
4. Sparse case 额外执行 cp.async.bulk.tensor 加载 metadata payload：DRAM -> LMEM。
5. TMEM_ALLOC 分配 A 和 D 的 TMEM 地址。
6. tcgen05.cp shape=4 将 A payload 从 LMEM 写入 A TMEM。
7. tcgen05.cp 将 C payload 从 LMEM 写入 D TMEM：Case 1 使用 shape=3 写入 1024B FP32 C，Case 2/3 使用 shape=4 写入 512B FP16 C。
8. Sparse case 额外执行 tcgen05.cp shape=1，将 64B metadata packet 写入 A TMEM payload 后的 metadata 位置。
9. 在 LMEM 构造 operand_block，携带 D taddr、A taddr、B shared-memory descriptor 和 C/D 格式。
10. 构造 idesc。Case 1 使用 F16F8；Case 2/3 使用 F8F6F4 并通过 TCU_MMA.sp 选择稀疏路径。
11. TCU_MMA 使用 enable-input-d 发射：dense case qualifier=0x1，sparse case qualifier=0x5。
12. TCU_MMA 发射后，core decode 只转发完整 RISC-V 指令字段；tcdecode 解析 idesc、operand_block 和 qualifier。
13. TensorUnit 从 TMEM 读 A payload，从 D TMEM 读 C payload；sparse case 同时读 A 侧 metadata packet；B payload 从 LMEM shared descriptor 读取。
14. TensorUnit 根据 metadata 对 compact A 和 dense B 做 sparse routing，再把 routed 8x8 primitive 和 C 输入送入 OpenTensorCore。
15. mbar_commit / mbarrier_arrive_token / mbarrier_wait 等待 MMA 完成。
16. 每个 lane 通过 8 次 TCU_LD 从 TMEM 读回 D 的 8 个 u32 word，并写入 DRAM 输出。
17. TMEM_DEALLOC 释放 D/A。

期望输出：

Case 1：

```text
D = A_fp16 * B_fp8 + C_fp32
```

由于 A 是 identity，D 应等于 B 的 FP8 有效值矩阵加 C，并以 FP32 输出。

Case 2：

```text
D = sparse_2_4(A_fp8) * B_fp8 + C_fp16
```

Case 3：

```text
D = sparse_1_4(A_fp8) * B_fp8 + C_fp16
```

三个 case 同时检查两组 host 侧 fp32 reference：

- golden1_quantized_input：A/B/C 先按目标输入格式量化，再用量化后的有效值做 fp32 A*B+C accumulate。
- golden2_raw_fp32_input：A/B/C 使用 host 生成的原始 fp32 数值直接做 fp32 A*B+C accumulate，不经过输入格式量化回读。

实际结果：

```text
case 1 tcgen05_dense_fp16_fp8_to_fp32:
  GOLDEN1_QUANTIZED_INPUT:
    max_abs_error=0
    mean_abs_error=0
    rms_error=0
    threshold=0.01
  GOLDEN2_RAW_FP32_INPUT:
    max_abs_error=0
    mean_abs_error=0
    rms_error=0
    threshold=0.01
  PASS

case 2 tcgen05_sparse_2_4_fp8_fp8_to_fp16:
  GOLDEN1_QUANTIZED_INPUT:
    max_abs_error=0
    mean_abs_error=0
    rms_error=0
    threshold=0.05
  GOLDEN2_RAW_FP32_INPUT:
    max_abs_error=0
    mean_abs_error=0
    rms_error=0
    threshold=0.05
  PASS

case 3 tcgen05_sparse_1_4_fp8_fp8_to_fp16:
  GOLDEN1_QUANTIZED_INPUT:
    max_abs_error=0
    mean_abs_error=0
    rms_error=0
    threshold=0.05
  GOLDEN2_RAW_FP32_INPUT:
    max_abs_error=0
    mean_abs_error=0
    rms_error=0
    threshold=0.05
  PASS

aggregate:
  instrs=39762
  cycles=29891
  IPC=1.330233
  PASS
```

### 3.3 tcu_ldst

测试目标：

验证 TMEM register path 的基本读写一致性，覆盖 TCU_ST、TCU_LD、TADDR lane 映射、wait_st 和 wait_ld。

测试输入：

- 每个 lane 生成一个 32-bit pattern：

```text
pattern[lane] = 0xCAFE0000 | lane
```

- 输出 DRAM 缓冲区初始填充为 0xDEADBEEF。

测试配置：

- 测试对象：Vortex SIMX + TensorUnit TMEM load/store path
- kernel：tests/regression/tcu_ldst
- grid：1x1
- block：32x1
- TMEM allocation：col_span=32
- 输出：32 个 u32，每个 lane 一个

测试指令序列：

1. TMEM_ALLOC 分配覆盖 32 lanes 的 TMEM 地址。
2. 每个 lane 执行 TCU_ST，将自己的 pattern 写入 TMEM。
3. TCU_WAIT_ST 等待 store 完成。
4. 每个 lane 执行 TCU_LD，从同一 warp-uniform TADDR 读回自己的 lane 数据。
5. TCU_WAIT_LD 等待读完成。
6. 每个 lane 将 observed 写入 DRAM out[lane]。
7. TMEM_DEALLOC 释放 TMEM。

期望输出：

```text
out[lane] = 0xCAFE0000 | lane
```

实际结果：

```text
instrs=23890
cycles=30777
IPC=0.776229
PASS
```

### 3.4 tcu_mbarrier_pending_tx

测试目标：

验证 cp.async.bulk.tensor 的 DRAM->LMEM functional copy 路径，以及 mbarrier::complete_tx::bytes 对 pending_tx 计数的驱动。该测试确认 mbarrier_wait 的放行依赖 arrival 和 tx bytes 两个条件。

测试输入：

- DRAM 输入：1024 个 u32，总 4096B。
- 输入模式：

```text
input[i] = 0xCAFE0000 | i
```

- 输出 DRAM：1 个 u32，初值 0xDEADBEEF。

测试配置：

- 测试对象：Vortex SIMX cp.async.bulk.tensor + mbarrier path
- kernel：tests/regression/tcu_mbarrier_pending_tx
- grid：1x1
- block：32x1
- tensor_map：128B rank-1 tensor_map，element_type=U32，box_size=1024，global_stride=4B，element_stride=1
- cpabulk_transfer_args：LMEM 32B 参数块，保存 LMEM 目标地址、mbarrier 地址、coords 全 0
- LMEM layout：8B mbarrier + 32B args + 4096B destination
- tx_bytes：4096

测试指令序列：

1. 在 LMEM 中构造 mbarrier_state 和 cpabulk_transfer_args。
2. mbarrier_init 初始化 barrier count=1。
3. mbarrier_expect_tx 设置期望传输字节数 4096。
4. cp.async.bulk.tensor.complete_tx 从 DRAM tensor_map 指向的源地址搬运 4096B 到 LMEM。
5. 搬运完成时触发 mbarrier_complete_tx(mbar_addr, 4096)。
6. mbarrier_arrive 消耗 arrival 条件。
7. mbarrier_wait 等待 phase 翻转；只有 arrival 和 complete_tx bytes 都满足后才放行。
8. kernel 读取 LMEM destination 的第一个 u32，并写回 DRAM 输出。

期望输出：

```text
observed = input[0] = 0xCAFE0000
```

实际结果：

```text
instrs=27026
cycles=33553
IPC=0.805472
observed first word=0xcafe0000
PASS
```

## 4. Coverage

### 4.1 已覆盖的 OTC standalone 功能

- Dense 16x16x16 MMA：覆盖 16x16 宏级矩阵乘到 8x8 TensorCoreTop primitive 的拆分和组合。
- FP8 输入路径：覆盖 FP8_E4M3 test vector 量化、FP8 到 FP9 的入口转换、TensorCoreTop FP9 乘加。
- FP16 输入路径：case 2 覆盖 A=FP16、B=FP8 的非对称输入转换。
- FP16 输出路径：case 1/3/4/5 覆盖 FP22 retire 结果转 FP16 输出后的误差检查。
- FP32 输出路径：case 2 覆盖 FP22 retire 结果转 FP32 输出。
- C 输入累加：case 1/3/4/5 覆盖 FP16 C 输入，case 2 覆盖 FP32 C 输入；所有 standalone case 的期望表达式都是 A*B+C。
- 多 K-slice 累加：case 3 覆盖 K=64 下多次 primitive 结果作为下一次 C 输入的 partial accumulation。
- 2:4 sparse metadata：case 4 覆盖每 4 个 K 元素保留 2 个非零值的 metadata 编码和 sparse_select routing。
- 1:4 sparse metadata：case 5 覆盖每 4 个 K 元素保留 1 个非零值的 metadata 编码、K=64 扩展和 sparse_select routing。
- fp32 golden reference：所有 OTC case 都使用格式量化后的有效输入值进行 fp32 reference 矩阵乘。

### 4.2 已覆盖的 Vortex+TC kernel 功能

- Core decode 到 TensorUnit dispatch：tcgen05_mma_minimal 覆盖 TCU_MMA 从 RISC-V kernel 发射到 TensorUnit 的端到端路径。
- tcdecode descriptor 解析：tcgen05_mma_minimal 覆盖 idesc 和 operand_block 的解析。
- tensor_map 描述符：tcgen05_mma_minimal 和 tcu_mbarrier_pending_tx 覆盖 128B tensor_map 被 kernel 传入并由 Cmodel 使用。
- cp.async.bulk.tensor DRAM->LMEM：tcgen05_mma_minimal 和 tcu_mbarrier_pending_tx 覆盖从 DRAM tensor_map 源地址到 LMEM 目标地址的搬运。
- cp.async.bulk.tensor complete_tx：tcu_mbarrier_pending_tx 覆盖 bytes 计数驱动 mbarrier pending_tx。
- LMEM->TMEM：tcgen05_mma_minimal 覆盖 tcgen05.cp 将 A payload 从 LMEM 写入 TMEM。
- C payload 初始化 D TMEM：tcgen05_mma_extended 覆盖 C payload 从 DRAM 经 LMEM 写入 D TMEM，并由 enable-input-d TCU_MMA 作为 C 输入参与 A*B+C。
- TMEM allocation：tcgen05_mma_minimal 和 tcu_ldst 覆盖 TMEM_ALLOC/TMEM_DEALLOC。
- TMEM register path：tcu_ldst 覆盖 TCU_ST 和 TCU_LD 的 round-trip。
- TADDR lane mapping：tcu_ldst 覆盖 warp-uniform TADDR 加 lane id 后访问 32 lanes。
- mbarrier wait：tcgen05_mma_minimal 覆盖 mbar_commit + mbarrier_wait 等待 MMA；tcu_mbarrier_pending_tx 覆盖 complete_tx bytes 条件。
- TCU_LD result readback：tcgen05_mma_minimal 覆盖通过寄存器路径从 TMEM 读出 D，再由普通 store 写回 DRAM。
- FP16*FP8->FP32 非对称 TCU_MMA：tcgen05_mma_extended case 1 覆盖 idesc.kind=F16F8、operand_block 中的 C/D 格式字段、tcdecode 解析和 TensorUnit 执行的完整 kernel 路径。
- TCU_MMA.sp 2:4 sparse：tcgen05_mma_extended case 2 覆盖 idesc.sparsity_kind=0、metadata payload 从 DRAM 经 LMEM 写入 A TMEM metadata 位置、TensorUnit 读取 metadata packet 和 sparse routing。
- TCU_MMA.sp 1:4 sparse：tcgen05_mma_extended case 3 覆盖 idesc.sparsity_kind=1、1:4 metadata line 解释、compact A payload 和 dense B payload 的 routed primitive 计算。
- Sparse metadata 搬运链路：tcgen05_mma_extended 的 sparse case 覆盖 metadata 从 descriptor 指向的 DRAM payload 到 LMEM，再通过 tcgen05.cp 写入 TMEM 后被 TensorUnit 消费的端到端路径。

### 4.3 当前未覆盖或只部分覆盖的内容

- OTC standalone 没有覆盖 Core/TensorUnit decode，也没有覆盖 RISC-V 指令编码；它只覆盖 TensorCoreTop primitive 级计算与 harness 级 sparse routing。
- OTC standalone 的 sparse case 当前覆盖 metadata 编码和 sparse_select routing，但不覆盖 MetaMem/BMem packet-level sparse source read 端口。
- OTC standalone 的非对称精度 case 覆盖 FP16/FP8 到 FP9 的输入转换和 FP32 输出，但 TensorCoreTop 本身仍消费 FP9 primitive；它不是 tcdecode 层面的 idesc 非对称精度测试。
- 带 C 的 OTC standalone case 使用非负输入。原因是当前 OTC final add 的异号 product+C 与普通 fp32 golden 对比会暴露额外数值偏差；本组测试将该问题排除在 coverage 外，专注 K 累加和 sparse metadata 功能。
- Vortex+TC kernel 已覆盖 FP16*FP8 非对称输入和 2:4/1:4 sparse 的单个 16x16x16 tile；仍未覆盖 K=64 多 K-tile 的完整 kernel 端到端累加路径。
- Sparse metadata 当前通过 A TMEM payload 后的 64B metadata packet 建模，未覆盖独立 TMEM meta_col 存储空间，因为当前 TMEM 配置没有启用独立 metadata columns。
- 当前 Vortex+TC kernel 不覆盖 TMEM->LMEM 反向 bulk copy；该路径已经不作为设计目标。TMEM 结果回读使用 TCU_LD 到寄存器，再由普通 store 写回内存。
- 当前 tcu_mbarrier_pending_tx 只验证 copy 目标首个 u32，未逐字节检查完整 4096B payload。

### 4.4 建议后续补充的测试

- Vortex kernel 版 K=64 多 K-tile MMA，覆盖真实 TCU_MMA 序列上的 D/partial 累加。
- Vortex kernel 版 K=32/K=64 sparse MMA，覆盖多 K-tile sparse metadata packet 选择和跨 tile 累加。
- cp.async.bulk.tensor 完整 payload 校验，把 tcu_mbarrier_pending_tx 从首字检查扩展为 4096B 全量比较。
- OTC final add 异号数值专项测试，用于定位和修正 product+C 异号情况下与 fp32 golden 的偏差。
