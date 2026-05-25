#pragma once

#include "open_tensorcore/tensor_compute/tensor_core_top.h"

namespace tensor_core_test_utils {

// Load one primitive into TensorCoreTop using the current standalone test
// configuration for precision metadata.
inline void load_inputs(TensorCoreTop* sim,
                        const uint16_t a[TensorCoreTop::M][TensorCoreTop::K],
                        const uint16_t b[TensorCoreTop::K][TensorCoreTop::N],
                        const uint32_t c[TensorCoreTop::M][TensorCoreTop::N]) {
  if (nullptr == sim) {
    return;
  }
  uint32_t zero_operand1[TensorCoreTop::M][TensorCoreTop::N] = {};
  uint32_t fp22_c[TensorCoreTop::M][TensorCoreTop::N] = {};
  TensorCoreMeta meta{};
  meta.valid = true;
  for (int i = 0; i < TensorCoreTop::M; ++i) {
    for (int j = 0; j < TensorCoreTop::N; ++j) {
      fp22_c[i][j] = convert_c_to_fp22(c[i][j], PREC_FP16);
    }
  }
  sim->push_uop(a, b, fp22_c, meta);
}

// Clear staged input without disturbing internal pipeline state.
inline void load_invalid(TensorCoreTop* sim) {
  if (nullptr == sim) {
    return;
  }
  sim->staged_valid = false;
  sim->input_loaded = false;
  sim->staged_meta = {};
}

// Tick once and report whether the primitive retired this cycle.
inline bool run(TensorCoreTop* sim) {
  if (nullptr == sim) {
    return false;
  }
  sim->tick(true);
  return sim->retired.valid;
}

// Double-precision reference for one 8x8 primitive dot product plus C.
inline void reference_matmul(const uint16_t a_fp9[8][8],
                             const uint16_t b_fp9[8][8],
                             const uint32_t c_fp22[8][8],
                             double d_fp[8][8],
                             RoundingMode rm = RNE) {
  (void)rm;
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      double acc = 0.0;
      for (int k = 0; k < 8; k++) {
        double a = fp9_to_double(a_fp9[i][k]);
        double b = fp9_to_double(b_fp9[k][j]);
        acc += a * b;
      }
      d_fp[i][j] = acc + fp22_to_double(c_fp22[i][j]);
    }
  }
}

} // namespace tensor_core_test_utils
