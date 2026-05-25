#pragma once

#include <cstdint>
#include <cstdlib>

#include "config_register.h"
#include "tensor_cfg.h"
#include "sparse_select.h"

// One-cycle operand selection stage in front of a 4-multiplier dot lane.
struct tc_select_pipe {
  static constexpr uint32_t kSourceK = 8;
  static constexpr uint32_t kMulK = 4;

  struct {
    uint32_t a_src[kSourceK] = {};
    uint32_t b_src[kSourceK] = {};
    uint32_t c_in = 0;
    uint32_t sparsity_kind = vortex::tensor::sparse_none;
    uint16_t sparse_row_meta = 0;
    bool input_valid = false;
    TensorCoreMeta meta = {};
  } select_input;

  struct {
    uint32_t a4[kMulK] = {};
    uint32_t b4[kMulK] = {};
    uint32_t c_in = 0;
    bool valid = false;
    TensorCoreMeta meta = {};
  } r1;

  void reset() {
    select_input = {};
    select_input.sparsity_kind = vortex::tensor::sparse_none;
    r1 = {};
  }

  bool out_valid() const {
    return r1.valid;
  }

  bool active() const {
    return select_input.input_valid || r1.valid;
  }

  bool in_ready(bool out_ready) const {
    return out_ready || !r1.valid;
  }

  const uint32_t& out_a(uint32_t lane) const {
    return r1.a4[lane];
  }

  const uint32_t& out_b(uint32_t lane) const {
    return r1.b4[lane];
  }

  uint32_t out_c() const {
    return r1.c_in;
  }

  const TensorCoreMeta& out_meta() const {
    return r1.meta;
  }

  void tick(bool out_ready) {
    const bool s1_ready = out_ready || !r1.valid;
    if (!s1_ready) {
      return;
    }

    if (!select_input.input_valid) {
      r1 = {};
      return;
    }

    switch (select_input.sparsity_kind) {
    case vortex::tensor::sparse_none:
      select_dense();
      break;
    case vortex::tensor::sparse_2_4:
      vortex::sparse::select_sparse_2_4_lane(select_input.sparse_row_meta,
                                             select_input.a_src,
                                             select_input.b_src,
                                             r1.a4,
                                             r1.b4);
      break;
    case vortex::tensor::sparse_1_4:
      vortex::sparse::select_sparse_1_4_lane(select_input.sparse_row_meta,
                                             select_input.a_src,
                                             select_input.b_src,
                                             r1.a4,
                                             r1.b4);
      break;
    default:
      std::abort();
    }

    r1.c_in = select_input.c_in;
    r1.valid = true;
    r1.meta = select_input.meta;
  }

private:
  void select_dense() {
    uint32_t base = (select_input.meta.accum_phase_id & 0x1u) * kMulK;
    for (uint32_t lane = 0; lane < kMulK; ++lane) {
      r1.a4[lane] = select_input.a_src[base + lane];
      r1.b4[lane] = select_input.b_src[base + lane];
    }
  }
};
