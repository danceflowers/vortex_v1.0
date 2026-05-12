#pragma once

// ============================================================================
// tensor_debug_utils.h —— TensorUnit 调试与日志（单实例简化版）
// ============================================================================

#include <iosfwd>

#include "tensor_mem_port_types.h"
#include "types.h"

namespace vortex {

op_string_t op_string(TcuType tcu_type, IntrTcuArgs args);

} // namespace vortex
