// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include "tmem.h"

namespace vortex {

// ============================================================================
// TensorUnit（执行侧）与 TmemSystem（存储侧）之间的端口级消息定义
//
// TensorUnit 内部的 TMEM↔SRAM 传输管线通过 SimPort<TensorMemPortReq/Rsp>
// 向 TmemSystem 发出逐包读写请求；TmemSystem 完成 TMEM 访问后返回响应。
// 每条请求/响应恰好对应 1 个 64B TMEM 数据包事务。
//
// TensorAsyncOpCompletion 在异步宏操作（mma_load / mma_store / wmma）的
// 所有微操作全部完成后由 TensorUnit 发出，通知 Core 解除等待。
//
// TmaRefillChunkReq/Rsp 由 TMEM_SHIFT 重填逻辑发出，经由 TMA 前端从外部
// DRAM 获取单个 64B 行块。
// ============================================================================

// 端口级请求：跨 TensorExecuteSystem ↔ TensorMemSystem 模块边界
// 载荷描述恰好一次 TMEM 包事务，加上共享入口/出口 FIFO 的仲裁年龄
struct TensorMemPortReq {
  enum class AccessType : uint8_t {
    Read = 0,
    Write,
  };

  uint64_t request_id = 0;          // 全局唯一请求 ID，用于匹配响应
  uint64_t arbitration_age = 0;     // 仲裁年龄：(async_id << 32 | packet_ordinal)，越小越优先
  AccessType access_type = AccessType::Read;  // 读/写类型
  Tmem::PortRequestDesc port_request = {};    // TMEM 端口请求描述符（handle, window, packet_idx）
  TmemPacket write_packet = {};               // 写请求载荷（仅 Write 时有效）
};

// 端口级响应：共享 TMEM FIFO 授权请求并完成实际 TMEM 包读写后返回
struct TensorMemPortRsp {
  uint64_t request_id = 0;          // 对应请求的 ID
  TensorMemPortReq::AccessType access_type = TensorMemPortReq::AccessType::Read;
  TmemPacket read_packet = {};      // 读响应载荷（仅 Read 时有效）
};

// 异步操作完成事件：当一条异步张量宏操作达到架构可见完成点时，由
// TensorExecuteSystem 发出。Core 侧的 async_tensor_waiters_ 监听此事件
// 以唤醒等待的 warp（TMA_WAIT / TC_FENCE）。
struct TensorAsyncOpCompletion {
  uint32_t async_id = 0;            // 对应的异步操作 ID
};

// TMEM_SHIFT 重填请求：由 TMEM 滑窗逻辑发出，经由 TMA 前端从外部行主序
// 存储器获取恰好 1 个 64B 行块。
struct TmaRefillChunkReq {
  uint64_t request_id = 0;          // 全局唯一请求 ID
  uint32_t descriptor_id = 0;       // TMA 描述符 ID（确定数据源地址布局）
  uint32_t chunk_idx = 0;           // 行块索引
  uint32_t row_bytes = 0;           // 行宽字节数
};

// TMEM_SHIFT 重填响应
struct TmaRefillChunkRsp {
  uint64_t request_id = 0;          // 对应请求的 ID
  bool success = false;             // 是否成功
  TmemPacket packet = {};           // 获取到的 64B 数据包
};

} // namespace vortex
