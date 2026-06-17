// TensorUnit debug and log-formatting helper implementation.

#include "tensor_debug_utils.h"

#include <cstdlib>
#include <iostream>
#include <string>

#include "tensor_cfg.h"

namespace vortex {

namespace vt = vortex::tensor;

op_string_t op_string(TcuType tcu_type, IntrTcuArgs args) {
  auto fmt_a = args.fmt_a;
  auto fmt_b = args.fmt_b;
  (void)fmt_a; (void)fmt_b;
  switch (tcu_type) {
  case TcuType::TMEM_ALLOC:
    return {"TMEM_ALLOC", ""};
  case TcuType::TMEM_REL_PERMIT:
    return {"TMEM_REL_PERMIT", ""};
  case TcuType::TMEM_SHIFT:
    return {"TMEM_SHIFT", ""};
  case TcuType::MBAR_INIT:
    return {"MBAR_INIT", ""};
  case TcuType::MBAR_ARRIVE:
    return {"MBAR_ARRIVE", ""};
  case TcuType::MBAR_WAIT:
    return {"MBAR_WAIT", ""};
  // ----- New tcgen05-aligned ISA -----
  case TcuType::TMEM_DEALLOC:
    return {"TMEM_DEALLOC", ""};
  case TcuType::TMEM_CP: {
    auto shape = static_cast<uint32_t>(args.cp_shape);
    auto decomp = static_cast<uint32_t>(args.cp_decompress);
    return {"TMEM_CP.shape" + std::to_string(shape) + ".decomp" + std::to_string(decomp), ""};
  }
  case TcuType::CPABULK_TENSOR_LD:
    return {"CPABULK_TENSOR_LD." + std::to_string(args.dim_count) + "d", ""};
  case TcuType::CPABULK_TENSOR_ST:
    return {"CPABULK_TENSOR_ST." + std::to_string(args.dim_count) + "d", ""};
  case TcuType::MBAR_FENCE:
    return {std::string("MBAR_FENCE.") + (args.fence_mode == TcuFenceMode::After ? "AFTER" : "BEFORE"), ""};
  case TcuType::MBAR_COMMIT:
    return {"MBAR_COMMIT", ""};
  case TcuType::MBAR_EXPECT_TX:
    return {"MBAR_EXPECT_TX", ""};
  case TcuType::MBAR_COMPLETE_TX:
    return {"MBAR_COMPLETE_TX", ""};
  case TcuType::MBAR_TEST_TRY_WAIT:
    return {std::string("MBAR_") + (args.test_or_try == TcuTestTryWait::Try ? "TRY_WAIT" : "TEST_WAIT"), ""};
  case TcuType::TCU_WMMA: {
    std::string suffix;
    if (args.ws) suffix += ".ws";
    if (args.sp) suffix += ".sp";
    if (args.enable_input_d) suffix += ".accum";
    return {"TCU_WMMA" + suffix, ""};
  }
  case TcuType::TCU_LD: {
    auto shape = static_cast<uint32_t>(args.ld_shape);
    return {"TCU_LD.shape" + std::to_string(shape) + ".num" + std::to_string(args.ld_num), ""};
  }
  case TcuType::TCU_ST: {
    auto shape = static_cast<uint32_t>(args.ld_shape);
    return {"TCU_ST.shape" + std::to_string(shape) + ".num" + std::to_string(args.ld_num), ""};
  }
  case TcuType::TCU_WAIT_LD:
    return {"TCU_WAIT_LD", ""};
  case TcuType::TCU_WAIT_ST:
    return {"TCU_WAIT_ST", ""};
  default:
    std::abort();
  }
}

} // namespace vortex
