#pragma once

// TensorUnit debug and log-formatting helpers.

#include <iosfwd>

#include "tensor_mem_port_types.h"
#include "types.h"

namespace vortex {

op_string_t op_string(TcuType tcu_type, IntrTcuArgs args);

} // namespace vortex
