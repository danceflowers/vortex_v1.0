# tensor_helper/test 架构说明书

## 1. 目录职责

`tensor_helper/test/` 是 `open_tensorcore` 的 standalone 测试入口目录。

它不参与 `simx` 主路径的时序推进，只服务于：

- 独立验证 `TensorCoreTop`
- reference 与实现对比
- OpenTensorCore ISA decode 示例

## 2. 文件说明

### [main.cpp](./main.cpp)

standalone 测试主程序。

主要内容：

- 随机矩阵生成
- reference 乘法/加法树结果生成
- 驱动 `TensorCoreTop`
- 比较 `fp22_out` 与 reference
- 打印输出矩阵和统计周期数

它是 `Makefile` 中当前默认编译的 standalone 测试源文件。

### [otc_decode.h](./otc_decode.h)

OpenTensorCore ISA decode 框架头文件。

内容包括：

- `OTC_OpType`
- `ExecUnit`
- `DecodedInst`
- `ISA_Entry`
- `OTC_Decoder`

定位：

- 更像测试/实验性 decode 框架
- 不是 `simx` 主控制路径的一部分

### [otc_decode.cpp](./otc_decode.cpp)

`OTC_Decoder` 的实现。

主要职责：

- 默认 ISA 表
- 指令字段提取
- 表驱动 decode

## 3. 维护建议

- 新增 standalone 验证和 directed test，优先放这里
- 若某个 decode 逻辑需要进入主路径，应重新评估后迁入 `simx` 正式 decode 目录，而不是直接复用这里
