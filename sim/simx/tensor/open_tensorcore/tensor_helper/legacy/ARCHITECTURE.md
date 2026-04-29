# tensor_helper/legacy 架构说明书

## 1. 目录职责

`tensor_helper/legacy/` 是历史遗留代码隔离区。

这里的文件有两个共同特点：

- 曾经参与过旧版 `open_tensorcore` 或旧版 `tensor_unit` 的实现
- 当前不应作为主路径源码继续依赖

本目录存在的意义是“保留参考”，不是“参与构建”。

## 2. 文件说明

### [con_to_fp9.h](./con_to_fp9.h)

旧版 `type -> fp9` 转换实现。

特点：

- 带有较多硬件化解释
- 但当前主路径已经由 `fp_types.h` 和 memory 模块的新接口接管

定位：

- 历史参考
- 不应重新接回主路径

### [fp22_to_fp8.h](./fp22_to_fp8.h)

旧版 `fp22 -> fp8` 转换实现。

当前主路径已经有更新后的转换工具链，这个文件仅作为旧实现参考保留。

### [tensor_unit.cpp.backup](./tensor_unit.cpp.backup)

旧版 `tensor_unit.cpp` 备份。

定位：

- 仅供人工对照重构前后差异
- 不应再作为实现来源

### [readme](./readme)

遗留空文件/占位文件，当前没有实际文档价值。

## 3. 维护建议

- 不要在这个目录新增“半启用”代码
- 若确认某个 legacy 文件已无参考价值，后续可以单独做清理
- 任何新逻辑都不应 include 这里的头文件
