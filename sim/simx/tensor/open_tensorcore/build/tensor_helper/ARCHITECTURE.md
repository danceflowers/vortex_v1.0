# build/tensor_helper 架构说明书

## 1. 目录职责

`build/tensor_helper/` 是 standalone 测试构建过程中自动生成的中间产物目录。

它镜像的是源目录中 `tensor_helper/` 相关编译单元的目标文件和依赖文件布局。

## 2. 当前文件说明

### [main.o](./main.o)

旧布局下 `tensor_helper/test/main.cpp` 编译得到的目标文件。

### [main.d](./main.d)

对应的依赖文件。

## 3. 使用原则

- 本目录不应手工维护
- 文件可由 `make clean` 清理并重新生成
