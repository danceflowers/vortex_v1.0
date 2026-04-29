# build 架构说明书

## 1. 目录职责

`build/` 是 `open_tensorcore` standalone 测试的构建产物目录，不是源码目录。

它的内容由 [../Makefile](../Makefile) 生成，主要用于：

- 存放 `tensor_helper/test/main.cpp` 的目标文件
- 存放依赖文件 `.d`
- 生成 standalone 可执行文件 `build/main`

## 2. 当前文件说明

### [main](./main)

当前 standalone 测试可执行文件。

### [main.o](./main.o)

主程序目标文件。

### [main.d](./main.d)

主程序依赖文件，供增量编译使用。

### [tensor_helper/main.o](./tensor_helper/main.o)
### [tensor_helper/main.d](./tensor_helper/main.d)

旧或中间构建布局下留下的目标/依赖文件。

## 3. 使用原则

- 本目录不应被当成源码目录维护
- 文档、实现、测试逻辑都不应新增到这里
- 若需要清理，使用 `make clean`
