# tensor_compute 架构说明书

## 1. 目录职责

`tensor_compute/` 是 `open_tensorcore` 的算术核心。

它实现的是：

- 浮点基础算子
- 乘法器/加法器流水
- 加法树
- 单元素 `tc_mul_add`
- 8x8 阵列顶层 `TensorCoreTop`

这个目录几乎全部是 header-only CModel。

## 2. 文件逐项说明

### [fp_types.h](./fp_types.h)

本目录最基础的公共浮点工具头。

主要内容：

- `PrecisionType`
- `RoundingMode`
- 各类格式转换
- `fp9 / fp22 / fp16 / fp32 / fp8` 的 pack/unpack
- `convert_to_fp9()`
- `convert_c_to_fp22()`
- `fp22_to_fp32()` 等工具
- rounding / clz / add helper

作用：

几乎所有 compute 和 local memory 模块都依赖这个文件。

### [fmul_s1.h](./fmul_s1.h)

乘法器第 1 级。

职责：

- 解析输入
- 处理特殊值
- 计算指数路径
- 生成尾数乘法前的中间信息

当前语义：

- 已按内部 `8/14` 算术域工作
- 但输入仍来自 `fp9`

### [fmul_s2.h](./fmul_s2.h)

乘法器第 2 级。

职责：

- 执行有效数字乘法
- 透传 `s1` 中间结果

### [fmul_s3.h](./fmul_s3.h)

乘法器第 3 级。

职责：

- 规格化
- 舍入
- 特殊值封装
- 输出打包后的浮点结果

### [tc_mul_pipe.h](./tc_mul_pipe.h)

三阶段乘法流水封装。

职责：

- 用 `r1/r2/r3` 明确建模三级乘法流水
- 提供 `valid/ready` 反压协议
- 把 `fmul_s1/s2/s3` 组合成一个可被上层复用的 pipe

### [fadd_s1.h](./fadd_s1.h)

加法器第 1 级。

职责：

- 路径选择
- 指数对齐
- near/far path 前处理

### [fadd_s2.h](./fadd_s2.h)

加法器第 2 级。

职责：

- 舍入
- 封装最终结果

### [tc_add_pipe.h](./tc_add_pipe.h)

两级浮点加法流水封装。

职责：

- 将 `fadd_s1 + fadd_s2` 组合成标准两级 pipe
- 为 `merge_add` 和 `final_add` 提供统一接口

### [tc_add4_pipe_fp22.h](./tc_add4_pipe_fp22.h)

四输入 `fp22` 加法树流水。

职责：

- 完成 4 路乘法结果的归约
- 当前结构按 `add4.v` 风格拆成 4 级

在 `tc_mul_add` 中用途：

- 两组 4 输入加法树分别归约 8 个乘积

### [tc_mul_add.h](./tc_mul_add.h)

单个输出元素对应的核心计算单元。

职责：

- 管理 8 个 `mul_pipe`
- 两个 `add4_pipe_fp22`
- 一个 `merge_add`
- 一个 `final_add`
- `operand1` 延迟线

语义上：

- 对应一个 `D[i][j]`
- 计算 `dot(A_row_i, B_col_j) + operand1_or_C`

### [tensor_core_top.h](./tensor_core_top.h)

8x8 TensorCore 顶层阵列。

职责：

- 持有 `64` 个 `tc_mul_add`
- 提供 `push_uop()`
- 在 `tick()` 中将输入广播给所有单元
- 汇聚 64 路输出形成 `TensorCoreRetire`

这是 `TensorUnit` 真正对接的计算顶层。

### [fp22_to_fp16.h](./fp22_to_fp16.h)

独立的 `fp22 -> fp16` 转换辅助头。

定位：

- 被 `CMem` dump 路径和 test/reference 路径复用

### [sparse_select.h](./sparse_select.h)

稀疏选择/压缩逻辑预留文件。

当前定位：

- 为未来 sparse datapath 预留接口
- 当前主 dense 路径不是由它驱动

## 3. 本目录内部层次

可以再分成四层：

1. 基础数据类型与转换
   - `fp_types.h`
   - `fp22_to_fp16.h`
2. 基础两级/三级算子
   - `fmul_s1/s2/s3`
   - `fadd_s1/s2`
3. 流水线封装
   - `tc_mul_pipe.h`
   - `tc_add_pipe.h`
   - `tc_add4_pipe_fp22.h`
4. 阵列级组合
   - `tc_mul_add.h`
   - `tensor_core_top.h`

## 4. 维护建议

- 新增位级算子，优先放在本目录
- 新增 standalone/reference/test 逻辑，不要再混进 `tensor_core_top.h`
- 对 `TensorCoreMeta` 的扩展要与 `tensor_control/config_register.h` 同步维护
