# tensor_helper 架构说明书

## 1. 目录职责

`tensor_helper/` 放的是非主时序路径代码。

这里的代码可以分为三类：

1. debug
2. standalone test helper
3. legacy 隔离区

设计目的很明确：

- 不让测试/调试/reference 重新混回 `tensor_top / tensor_control / local_memory / tensor_compute`

## 2. 当前文件说明

### [tensor_debug_utils.h](./tensor_debug_utils.h)

调试工具接口。

主要提供：

- `op_string()`
- `log_window_plan_summary()`
- `dump_tensor_unit_state()`

### [tensor_debug_utils.cpp](./tensor_debug_utils.cpp)

调试工具实现。

作用：

- 把 tensor 指令格式化成可读字符串
- 打印 TMEM window 计划摘要
- 导出 `TensorUnit` 内部完整状态，便于死锁和流水问题定位

### [tensor_core_test_utils.h](./tensor_core_test_utils.h)

`TensorCoreTop` standalone helper。

作用：

- 生成/加载测试输入
- 驱动 `TensorCoreTop`
- 提供 `reference_matmul()` 之类的参考路径

说明：

这类代码不进入 `simx` 主时序路径，只服务于 `open_tensorcore` 单测。

### [tensor_mem_test_utils.h](./tensor_mem_test_utils.h)

本地存储 test helper。

作用：

- 批量填充 `AMem/BMem/CMem/MetaMem`
- 批量 dump `CMem`
- 为 reference 和独立测试提供便捷入口

这部分之前曾混在 memory 类中，现在已经外提到 helper 目录。

## 3. 子目录说明

### [test/](./test/ARCHITECTURE.md)

standalone 测试与 decode 示例。

### [legacy/](./legacy/ARCHITECTURE.md)

旧代码隔离区，不应参与主路径构建和新功能开发。

## 4. 维护建议

- 新的 debug dump / test scaffold / golden reference，优先放这里
- 如果一个 helper 会被主路径调用，要谨慎；只有纯日志类 helper 才适合被主路径直接 include
