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

#include <tensor_cfg.h>
#include <vx_intrinsics.h>

namespace vortex {
namespace tensor {

enum mem_layout {
  row_major,
  col_major
};

enum tcu_target : uint8_t {
  tcu_target_none = 0,
  tcu_target_a = 1,
  tcu_target_b = 2,
  tcu_target_c = 3,
};

// TMA descriptors describe which logical TMEM window a payload belongs to.
// This is distinct from MMA_LOAD.target, which selects the local fill destination.
enum tma_tile_role : uint8_t {
  tma_tile_role_none = 0,
  tma_tile_role_a = 1,
  tma_tile_role_b = 2,
  tma_tile_role_c = 3,
  tma_tile_role_d = 4,
};

static constexpr uint32_t max_operand_slots = 2;
static constexpr uint32_t tcu_tmem_op_ctl_window_bits = 8;
static constexpr uint32_t tcu_tmem_op_ctl_desc_bits = 16;
static constexpr uint32_t tcu_tmem_op_ctl_flags_bits = 8;
static constexpr uint32_t tcu_tmem_op_ctl_window_shift = 0;
static constexpr uint32_t tcu_tmem_op_ctl_desc_shift =
    tcu_tmem_op_ctl_window_shift + tcu_tmem_op_ctl_window_bits;
static constexpr uint32_t tcu_tmem_op_ctl_flags_shift =
    tcu_tmem_op_ctl_desc_shift + tcu_tmem_op_ctl_desc_bits;
static constexpr uint32_t tcu_tmem_op_flag_refill = 0x1u;
static constexpr uint32_t wmma_slot_ctl_slot_bits = 2;
static constexpr uint32_t wmma_slot_ctl_a_shift = 0;
static constexpr uint32_t wmma_slot_ctl_b_shift = wmma_slot_ctl_a_shift + wmma_slot_ctl_slot_bits;

inline __attribute__((always_inline)) constexpr uint32_t encode_wmma_slot_control(uint32_t a_slot_id,
                                                                                   uint32_t b_slot_id) {
  return ((a_slot_id & ((1u << wmma_slot_ctl_slot_bits) - 1)) << wmma_slot_ctl_a_shift)
       | ((b_slot_id & ((1u << wmma_slot_ctl_slot_bits) - 1)) << wmma_slot_ctl_b_shift);
}

enum tcu_payload_kind : uint8_t {
  tcu_payload_dense = 0,
  tcu_payload_sparse_payload = 1,
  tcu_payload_sparse_meta = 2,
};

enum tcu_sparse_mode : uint8_t {
  tcu_sparse_none = sparse_none,
  tcu_sparse_2_4 = sparse_2_4,
  tcu_sparse_1_4 = sparse_1_4,
};

struct tma_descriptor_t {
  uint64_t addr;
  uint32_t size_bytes;
  uint32_t stride_bytes;
  uint16_t rows;
  uint16_t cols;
  uint16_t elem_bytes;
  uint16_t flags;
  uint64_t meta_addr;
  uint32_t meta_size_bytes;
  uint16_t tmem_base;
  uint16_t meta_tmem_base;
  uint16_t bank_span;
  uint16_t meta_col_span;
  uint8_t tile_role;
  uint8_t payload_kind;
  uint8_t reserved[2];
} __attribute__((packed));

struct mma_descriptor_t {
  uint32_t fmt_a;
  uint32_t fmt_b;
  uint32_t fmt_c;
  uint32_t fmt_d;
  uint8_t ws;
  uint8_t sp;
  uint8_t sparse_mode;
  uint8_t transpose_a;
  uint8_t transpose_b;
  uint8_t reserved[3];
  uint16_t a_rows;
  uint16_t a_cols;
  uint16_t b_rows;
  uint16_t b_cols;
  uint16_t c_rows;
  uint16_t c_cols;
} __attribute__((packed));

template <typename At, typename Bt, typename Ct, typename Dt = Ct>
inline __attribute__((always_inline)) constexpr mma_descriptor_t make_mma_descriptor(uint8_t ws = 0,
                                                                                     uint8_t sp = 0,
                                                                                     uint8_t sparse_mode = 0,
                                                                                     uint16_t a_rows = 0,
                                                                                     uint16_t a_cols = 0,
                                                                                     uint16_t b_rows = 0,
                                                                                     uint16_t b_cols = 0,
                                                                                     uint16_t c_rows = 0,
                                                                                     uint16_t c_cols = 0,
                                                                                     uint8_t transpose_a = 0,
                                                                                     uint8_t transpose_b = 0) {
  return mma_descriptor_t{At::id, Bt::id, Ct::id, Dt::id,
                          ws, sp, sparse_mode, transpose_a, transpose_b, {0, 0, 0},
                          a_rows, a_cols, b_rows, b_cols, c_rows, c_cols};
}

inline __attribute__((always_inline)) uint16_t tmem_handle_base(uint32_t handle) {
  return handle & 0xff;
}

inline __attribute__((always_inline)) uint16_t tmem_handle_span(uint32_t handle) {
  return (handle >> 8) & 0xff;
}

static constexpr uint32_t mma_mem_ctl_target_bits = 2;
static constexpr uint32_t mma_mem_ctl_window_bits = 8;
static constexpr uint32_t mma_mem_ctl_tile_bits = 16;
static constexpr uint32_t mma_mem_ctl_slot_bits = 2;
static constexpr uint32_t mma_mem_ctl_target_shift = 0;
static constexpr uint32_t mma_mem_ctl_window_shift = mma_mem_ctl_target_shift + mma_mem_ctl_target_bits;
static constexpr uint32_t mma_mem_ctl_tile_shift = mma_mem_ctl_window_shift + mma_mem_ctl_window_bits;
static constexpr uint32_t mma_mem_ctl_slot_shift = mma_mem_ctl_tile_shift + mma_mem_ctl_tile_bits;

inline __attribute__((always_inline)) constexpr uint32_t mma_mem_ctl_mask(uint32_t bits) {
  return (bits >= 32) ? 0xffffffffu : ((1u << bits) - 1);
}

inline __attribute__((always_inline)) constexpr uint32_t encode_tmem_op_control(uint32_t window_id,
                                                                                 uint32_t desc_id = 0,
                                                                                 uint32_t flags = 0) {
  return ((window_id & mma_mem_ctl_mask(tcu_tmem_op_ctl_window_bits)) << tcu_tmem_op_ctl_window_shift)
       | ((desc_id & mma_mem_ctl_mask(tcu_tmem_op_ctl_desc_bits)) << tcu_tmem_op_ctl_desc_shift)
       | ((flags & mma_mem_ctl_mask(tcu_tmem_op_ctl_flags_bits)) << tcu_tmem_op_ctl_flags_shift);
}

inline __attribute__((always_inline)) constexpr uint32_t encode_mma_mem_control(tcu_target target,
                                                                                uint32_t slot_id,
                                                                                uint32_t window_id,
                                                                                uint32_t tile_id) {
  return ((static_cast<uint32_t>(target) & mma_mem_ctl_mask(mma_mem_ctl_target_bits)) << mma_mem_ctl_target_shift)
       | ((window_id & mma_mem_ctl_mask(mma_mem_ctl_window_bits)) << mma_mem_ctl_window_shift)
       | ((tile_id & mma_mem_ctl_mask(mma_mem_ctl_tile_bits)) << mma_mem_ctl_tile_shift)
       | ((slot_id & mma_mem_ctl_mask(mma_mem_ctl_slot_bits)) << mma_mem_ctl_slot_shift);
}

inline __attribute__((always_inline)) void bind_tmem_payload_region(tma_descriptor_t* desc, uint32_t handle) {
  desc->tmem_base = tmem_handle_base(handle);
  desc->bank_span = tmem_handle_span(handle);
}

inline __attribute__((always_inline)) void bind_tmem_meta_region(tma_descriptor_t* desc, uint32_t handle) {
  desc->meta_tmem_base = tmem_handle_base(handle);
  desc->meta_col_span = tmem_handle_span(handle);
}

inline __attribute__((always_inline)) uint32_t tmem_alloc(uint32_t bank_span) {
  uint32_t handle;
  __asm__ volatile (".insn r %2, 1, 2, %0, %1, x0"
    : "=r"(handle)
    : "r"(bank_span), "i"(RISCV_CUSTOM0)
    : "memory");
  return handle;
}

inline __attribute__((always_inline)) void tmem_free(uint32_t handle) {
  __asm__ volatile (".insn r %1, 2, 2, x0, %0, x0"
    :
    : "r"(handle), "i"(RISCV_CUSTOM0)
    : "memory");
}

inline __attribute__((always_inline)) void tmem_rel_permit() {
  __asm__ volatile (".insn r %0, 0, 3, x0, x0, x0"
    :
    : "i"(RISCV_CUSTOM0)
    : "memory");
}

inline __attribute__((always_inline)) uint32_t tma_load_ctl(uint32_t handle, uint32_t control) {
  uint32_t async_id;
  __asm__ volatile (".insn r %3, 3, 2, %0, %1, %2"
    : "=r"(async_id)
    : "r"(handle), "r"(control), "i"(RISCV_CUSTOM0)
    : "memory");
  return async_id;
}

inline __attribute__((always_inline)) uint32_t tma_load(uint32_t handle, uint32_t desc_id) {
  return tma_load_ctl(handle, encode_tmem_op_control(0, desc_id));
}

inline __attribute__((always_inline)) uint32_t tma_load(uint32_t handle, uint32_t desc_id, uint32_t window_id) {
  return tma_load_ctl(handle, encode_tmem_op_control(window_id, desc_id));
}

inline __attribute__((always_inline)) uint32_t tma_store_ctl(uint32_t handle, uint32_t control) {
  uint32_t async_id;
  __asm__ volatile (".insn r %3, 4, 2, %0, %1, %2"
    : "=r"(async_id)
    : "r"(handle), "r"(control), "i"(RISCV_CUSTOM0)
    : "memory");
  return async_id;
}

inline __attribute__((always_inline)) uint32_t tma_store(uint32_t handle, uint32_t desc_id) {
  return tma_store_ctl(handle, encode_tmem_op_control(0, desc_id));
}

inline __attribute__((always_inline)) uint32_t tma_store(uint32_t handle, uint32_t desc_id, uint32_t window_id) {
  return tma_store_ctl(handle, encode_tmem_op_control(window_id, desc_id));
}

inline __attribute__((always_inline)) void tma_wait(uint32_t async_id) {
  __asm__ volatile (".insn r %1, 7, 2, x0, %0, x0"
    :
    : "r"(async_id), "i"(RISCV_CUSTOM0)
    : "memory");
}

inline __attribute__((always_inline)) uint32_t tc_commit(uint32_t barrier_id) {
  uint32_t committed;
  __asm__ volatile (".insn r %2, 1, 3, %0, %1, x0"
    : "=r"(committed)
    : "r"(barrier_id), "i"(RISCV_CUSTOM0)
    : "memory");
  return committed;
}

inline __attribute__((always_inline)) void tc_fence_before() {
  __asm__ volatile (".insn r %0, 2, 3, x0, x0, x0"
    :
    : "i"(RISCV_CUSTOM0)
    : "memory");
}

inline __attribute__((always_inline)) void tc_fence_after() {
  __asm__ volatile (".insn r %0, 2, 3, x1, x0, x0"
    :
    : "i"(RISCV_CUSTOM0)
    : "memory");
}

inline __attribute__((always_inline)) void tc_fence() {
  tc_fence_before();
}

inline __attribute__((always_inline)) void tc_wait() {
  __asm__ volatile (".insn r %0, 7, 3, x0, x0, x0"
    :
    : "i"(RISCV_CUSTOM0)
    : "memory");
}

inline __attribute__((always_inline)) uint32_t tmem_shift_ctl(uint32_t handle, uint32_t control) {
  uint32_t async_id;
  __asm__ volatile (".insn r %3, 3, 3, %0, %1, %2"
    : "=r"(async_id)
    : "r"(handle), "r"(control), "i"(RISCV_CUSTOM0)
    : "memory");
  return async_id;
}

inline __attribute__((always_inline)) uint32_t tmem_shift(uint32_t handle) {
  return tmem_shift_ctl(handle, encode_tmem_op_control(0));
}

inline __attribute__((always_inline)) uint32_t tmem_shift(uint32_t handle, uint32_t window_id) {
  return tmem_shift_ctl(handle, encode_tmem_op_control(window_id));
}

inline __attribute__((always_inline)) uint32_t tmem_shift_refill(uint32_t handle, uint32_t refill_desc_id) {
  return tmem_shift_ctl(handle, encode_tmem_op_control(0, refill_desc_id, tcu_tmem_op_flag_refill));
}

inline __attribute__((always_inline)) uint32_t tmem_shift_refill(uint32_t handle, uint32_t window_id, uint32_t refill_desc_id) {
  return tmem_shift_ctl(handle, encode_tmem_op_control(window_id, refill_desc_id, tcu_tmem_op_flag_refill));
}

inline __attribute__((always_inline)) void mbarrier_init(uint32_t barrier_id, uint32_t count) {
  __asm__ volatile (".insn r %2, 4, 3, x0, %0, %1"
    :
    : "r"(barrier_id), "r"(count), "i"(RISCV_CUSTOM0)
    : "memory");
}

inline __attribute__((always_inline)) void mbarrier_arrive(uint32_t barrier_id) {
  __asm__ volatile (".insn r %1, 5, 3, x0, %0, x0"
    :
    : "r"(barrier_id), "i"(RISCV_CUSTOM0)
    : "memory");
}

inline __attribute__((always_inline)) void mbarrier_wait(uint32_t barrier_id) {
  __asm__ volatile (".insn r %1, 6, 3, x0, %0, x0"
    :
    : "r"(barrier_id), "i"(RISCV_CUSTOM0)
    : "memory");
}

template <uint32_t DescId>
inline __attribute__((always_inline)) void mma_load_mem(uint32_t handle,
                                                        uint32_t control) {
  static_assert(DescId < max_static_descriptor_id, "desc_id must fit in the encoded desc_id field");
  __asm__ volatile (".insn r %[insn], 5, 4, x%[desc_id], %[handle], %[control]"
    :
    : [insn]"i"(RISCV_CUSTOM0), [desc_id]"i"(DescId), [handle]"r"(handle), [control]"r"(control)
    : "memory");
}

template <uint32_t DescId>
inline __attribute__((always_inline)) void mma_load_a_slot(uint32_t handle,
                                                           uint32_t window_id,
                                                           uint32_t tile_id,
                                                           uint32_t slot_id) {
  if (slot_id >= max_operand_slots) {
    __builtin_trap();
  }
  auto control = encode_mma_mem_control(tcu_target_a, slot_id, window_id, tile_id);
  mma_load_mem<DescId>(handle, control);
}

template <uint32_t DescId>
inline __attribute__((always_inline)) void mma_load_b_slot(uint32_t handle,
                                                           uint32_t window_id,
                                                           uint32_t tile_id,
                                                           uint32_t slot_id) {
  if (slot_id >= max_operand_slots) {
    __builtin_trap();
  }
  auto control = encode_mma_mem_control(tcu_target_b, slot_id, window_id, tile_id);
  mma_load_mem<DescId>(handle, control);
}

template <uint32_t DescId>
inline __attribute__((always_inline)) void mma_load_c_slot(uint32_t handle,
                                                           uint32_t window_id,
                                                           uint32_t tile_id,
                                                           uint32_t slot_id) {
  if (slot_id >= max_operand_slots) {
    __builtin_trap();
  }
  auto control = encode_mma_mem_control(tcu_target_c, slot_id, window_id, tile_id);
  mma_load_mem<DescId>(handle, control);
}

template <uint32_t DescId>
inline __attribute__((always_inline)) void mma_store_mem(uint32_t handle,
                                                         uint32_t control) {
  static_assert(DescId < max_static_descriptor_id, "desc_id must fit in the encoded desc_id field");
  __asm__ volatile (".insn r %[insn], 6, 4, x%[desc_id], %[handle], %[control]"
    :
    : [insn]"i"(RISCV_CUSTOM0), [desc_id]"i"(DescId), [handle]"r"(handle), [control]"r"(control)
    : "memory");
}

template <uint32_t DescId>
inline __attribute__((always_inline)) void mma_store_c_slot(uint32_t handle,
                                                            uint32_t window_id,
                                                            uint32_t tile_id,
                                                            uint32_t slot_id) {
  if (slot_id >= max_operand_slots) {
    __builtin_trap();
  }
  auto control = encode_mma_mem_control(tcu_target_c, slot_id, window_id, tile_id);
  mma_store_mem<DescId>(handle, control);
}

namespace detail {

  template <typename F, std::size_t... Is>
  __attribute__((always_inline))
  constexpr void unroll_for_impl(std::index_sequence<Is...>, F&& f) {
    (f(std::integral_constant<std::size_t, Is>{}), ...);
  }

  template <std::size_t N, typename F>
  __attribute__((always_inline))
  constexpr void unroll_for(F&& f) {
    unroll_for_impl(std::make_index_sequence<N>{}, std::forward<F>(f));
  }

  template <typename T>
  struct raw_unsigned {
    using type = std::conditional_t<(sizeof(T) == 1), uint8_t,
      std::conditional_t<(sizeof(T) == 2), uint16_t,
        std::conditional_t<(sizeof(T) == 4), uint32_t,
          uint64_t>>>;
  };
  template <typename T>
  using raw_unsigned_t = typename raw_unsigned<T>::type;

  template <typename T, typename D>
  struct data_accessor_t {
    using Type = typename T::dtype;

    static inline D bit_fill(Type src) {
      static_assert(sizeof(D) % sizeof(Type) == 0, "D must be a multiple of Type in size");
      if constexpr (std::is_same_v<Type, D>) {
        return src; // passthrough
      } else {
        constexpr uint32_t count = sizeof(D) / sizeof(Type);
        constexpr uint32_t bits = 8 * sizeof(Type);
        using US = raw_unsigned_t<Type>;
        using UD = raw_unsigned_t<D>;
        auto src_u = *reinterpret_cast<const US*>(&src); // unsigned cast
        auto src_d = static_cast<UD>(src_u); // zero-extend
        UD result_u(0);
        for (uint32_t i = 0; i < count; i++) {
          result_u |= (src_d << (i * bits));
        }
        return *reinterpret_cast<const D*>(&result_u);
      }
    }

    static inline D pack_row(const Type *base, uint32_t ldm) {
      static_assert(sizeof(D) % sizeof(Type) == 0, "D must be a multiple of Type in size");
      constexpr uint32_t count = sizeof(D) / sizeof(Type);
      constexpr uint32_t bits = 8 * sizeof(Type);
      using US = raw_unsigned_t<Type>;
      using UD = raw_unsigned_t<D>;
      UD result_u(0);
      for (uint32_t i = 0; i < count; ++i) {
        auto src_u = *reinterpret_cast<const US*>(base); // unsigned cast
        auto src_d = static_cast<UD>(src_u); // zero-extend
        result_u |= (src_d << (i * bits));
        base += ldm; // next row
      }
      return *reinterpret_cast<const D*>(&result_u);
    }
  };

  template <typename D>
  struct data_accessor_t<int4, D> {

    static inline D bit_fill(uint8_t src) {
      constexpr uint32_t count = sizeof(D);
      assert((src & 0xf0) == 0 && "src must be a 4-bit value");
      using UD = raw_unsigned_t<D>;
      uint8_t src_u8 = (src << 4) | src; // pack 2 nibbles
      auto src_d = static_cast<UD>(src_u8); // zero-extend
      UD result_u(0);
      for (uint32_t i = 0; i < count; i++) {
        result_u |= (src_d << (i * 8));
      }
      return *reinterpret_cast<const D*>(&result_u);
    }
  };

  template <typename D>
  struct data_accessor_t<uint4, D> {

    static inline D bit_fill(uint8_t src) {
      constexpr uint32_t count = sizeof(D);
      assert((src & 0xf0) == 0 && "src must be a 4-bit value");
      using UD = raw_unsigned_t<D>;
      uint8_t src_u8 = (src << 4) | src; // pack 2 nibbles
      auto src_d = static_cast<UD>(src_u8); // zero-extend
      UD result_u(0);
      for (uint32_t i = 0; i < count; i++) {
        result_u |= (src_d << (i * 8));
      }
      return *reinterpret_cast<const D*>(&result_u);
    }
  };
}

template <uint32_t NT, // number of threads per warp
          typename At, // input type A
          typename Bt, // input type B
          typename Ot> // output type (C,D)
struct wmma_context_ab {
private:
  using cfg = wmma_ab_config_t<NT, At, Bt, Ot>;

  enum frag_use_t { matrix_a, matrix_b, accumulator };

  using vreg_t = float;

  template <frag_use_t U, typename T, uint32_t N>
  struct fragment_t {
    using Type = T;
    static constexpr frag_use_t Use = U;
    static constexpr uint32_t NR = N;
    std::array<vreg_t, N> data;
  };

public:

  using input_a_t = typename At::dtype;
  using input_b_t = typename Bt::dtype;
  using output_t = typename Ot::dtype;

  using input_a_accessor_t = detail::data_accessor_t<At, vreg_t>;
  using input_b_accessor_t = detail::data_accessor_t<Bt, vreg_t>;
  using output_acessor_t = detail::data_accessor_t<Ot, vreg_t>;

  static constexpr uint32_t input_a_is_subbyte = (At::bits < 8);
  static constexpr uint32_t input_b_is_subbyte = (Bt::bits < 8);

  static constexpr uint32_t a_i_ratio = cfg::a_i_ratio;
  static constexpr uint32_t b_i_ratio = cfg::b_i_ratio;
  static constexpr uint32_t tileM = cfg::tileM;
  static constexpr uint32_t tileN = cfg::tileN;
  static constexpr uint32_t tileK_a = cfg::tileK_a;
  static constexpr uint32_t tileK_b = cfg::tileK_b;

  using fragment_a   = fragment_t<matrix_a, input_a_t, cfg::NRA>;
  using fragment_b   = fragment_t<matrix_b, input_b_t, cfg::NRB>;
  using fragment_acc = fragment_t<accumulator, output_t, cfg::NRC>;

  template <typename Frag, typename T>
  static __attribute__((always_inline)) void fill_fragment(Frag &dst, T value) {
    vreg_t fill_data;
    if constexpr (Frag::Use == accumulator) {
      fill_data = output_acessor_t::bit_fill(value);
    } else if constexpr (Frag::Use == matrix_a) {
      fill_data = input_a_accessor_t::bit_fill(value);
    } else {
      fill_data = input_b_accessor_t::bit_fill(value);
    }
    detail::unroll_for<Frag::NR>([&](auto r) {
      vreg_t tmp;
      __asm__ volatile("fmv.s %0, %1" : "=f"(tmp): "f"(fill_data));
      dst.data[r] = tmp;
    });
  }

  template <mem_layout src_layout = row_major, typename Frag>
  static __attribute__((always_inline)) void load_matrix_sync(Frag &dst, const void *src, size_t ldm) {
    uint32_t lane = vx_thread_id();
    if constexpr (Frag::Use == matrix_a) {
      // Load row-major matrix A
      uint32_t block_idx = (cfg::a_block_size == NT) ? 0 : (lane / cfg::a_block_size);
      uint32_t lane_in_blk = (cfg::a_block_size == NT) ? lane : (lane % cfg::a_block_size);
      uint32_t block_row = (lane_in_blk / cfg::tcK) + (block_idx * cfg::tcM);
      uint32_t block_col = (lane_in_blk % cfg::tcK) * a_i_ratio;
      uint32_t m_stride  = cfg::a_sub_blocks * cfg::tcM;
      uint32_t k_stride  = cfg::tcK * a_i_ratio;
      if constexpr (src_layout == col_major) {
        std::swap(block_row, block_col);
      }
      auto base = reinterpret_cast<const input_a_t*>(src) + block_row * ldm + block_col;
      detail::unroll_for<Frag::NR>([&](auto r) {
        uint32_t block_m  = r / cfg::k_steps;
        uint32_t block_k  = r % cfg::k_steps;
        uint32_t elem_row = block_m * m_stride;
        uint32_t elem_col = block_k * k_stride;
        if constexpr (src_layout == col_major) {
          static_assert(input_a_is_subbyte == false, "col_major layout is not supported for sub-byte matrix_a");
          std::swap(elem_row, elem_col);
          auto ptr = base + elem_row * ldm + elem_col;
          if constexpr (sizeof(vreg_t) == sizeof(input_a_t) && !input_a_is_subbyte) {
            dst.data[r] = *reinterpret_cast<const vreg_t*>(ptr);
          } else {
            dst.data[r] = input_a_accessor_t::pack_row(ptr, ldm);
          }
        } else {
          // raw_major layout
          auto ptr = base + elem_row * ldm + elem_col;
          assert(reinterpret_cast<uintptr_t>(ptr) % alignof(vreg_t) == 0 && "pointer must be aligned to 4 bytes");
          dst.data[r] = *reinterpret_cast<const vreg_t *>(ptr);
        }
      });
    } else if constexpr (Frag::Use == matrix_b) {
      // Load column-major matrix B
      uint32_t block_idx = (cfg::b_block_size == NT) ? 0 : (lane / cfg::b_block_size);
      uint32_t lane_in_blk = (cfg::b_block_size == NT) ? lane : (lane % cfg::b_block_size);
      uint32_t block_col = (lane_in_blk / cfg::tcK) + (block_idx * cfg::tcN);
      uint32_t block_row = (lane_in_blk % cfg::tcK) * b_i_ratio;
      uint32_t n_stride  = cfg::b_sub_blocks * cfg::tcN;
      uint32_t k_stride  = cfg::tcK * b_i_ratio;
      if constexpr (src_layout == col_major) {
        std::swap(block_row, block_col);
      }
      auto base = reinterpret_cast<const input_b_t*>(src) + block_row * ldm + block_col;
      detail::unroll_for<Frag::NR>([&](auto r) {
        uint32_t block_k = r / cfg::b_sub_steps;
        uint32_t block_n = r % cfg::b_sub_steps;
        uint32_t elem_row = block_k * k_stride;
        uint32_t elem_col = block_n * n_stride;
        if constexpr (src_layout == row_major) {
          static_assert(input_b_is_subbyte == false, "row_major layout is not supported for sub-byte matrix_b");
          auto ptr = base + elem_row * ldm + elem_col;
          if constexpr (sizeof(vreg_t) == sizeof(input_b_t) && !input_b_is_subbyte) {
            dst.data[r] = *reinterpret_cast<const vreg_t*>(ptr);
          } else {
            dst.data[r] = input_b_accessor_t::pack_row(ptr, ldm);
          }
        } else {
          // col_major layout
          std::swap(elem_row, elem_col);
          auto ptr = base + elem_row * ldm + elem_col;
          assert(reinterpret_cast<uintptr_t>(ptr) % alignof(vreg_t) == 0 && "pointer must be aligned to 4 bytes");
          dst.data[r] = *reinterpret_cast<const vreg_t *>(ptr);
        }
      });
    } else {
      // Load accumulator matrix C
      if constexpr (std::is_same_v<output_t, uint16_t>) {
        // FP16 accumulator is packed as 2x16-bit values in one 32-bit vreg.
        uint32_t tcN_pairs = cfg::tcN / 2;
        uint32_t block_row = lane / tcN_pairs;
        uint32_t block_col = (lane % tcN_pairs) * 2;
        uint32_t m_stride = cfg::tcM;
        uint32_t n_stride = cfg::tcN;
        if constexpr (src_layout == col_major) {
          std::swap(block_row, block_col);
        }
        auto base = reinterpret_cast<const output_t*>(src) + block_row * ldm + block_col;
        detail::unroll_for<Frag::NR>([&](auto r) {
          uint32_t block_m  = r / cfg::n_steps;
          uint32_t block_n  = r % cfg::n_steps;
          uint32_t elem_row = block_m * m_stride;
          uint32_t elem_col = block_n * n_stride;
          if constexpr (src_layout == col_major) {
            std::swap(elem_row, elem_col);
          }
          auto ptr = base + elem_row * ldm + elem_col;
          uint32_t packed = static_cast<uint32_t>(ptr[0]) | (static_cast<uint32_t>(ptr[1]) << 16);
          dst.data[r] = *reinterpret_cast<const vreg_t*>(&packed);
        });
      } else {
        uint32_t block_row = lane / cfg::tcN;
        uint32_t block_col = lane % cfg::tcN;
        uint32_t m_stride = cfg::tcM;
        uint32_t n_stride = cfg::tcN;
        if constexpr (src_layout == col_major) {
          std::swap(block_row, block_col);
        }
        auto base = reinterpret_cast<const output_t*>(src) + block_row * ldm + block_col;
        detail::unroll_for<Frag::NR>([&](auto r) {
          uint32_t block_m  = r / cfg::n_steps;
          uint32_t block_n  = r % cfg::n_steps;
          uint32_t elem_row = block_m * m_stride;
          uint32_t elem_col = block_n * n_stride;
          if constexpr (src_layout == col_major) {
            std::swap(elem_row, elem_col);
          }
          auto ptr = base + elem_row * ldm + elem_col;
          if constexpr (sizeof(vreg_t) == sizeof(output_t)) {
            dst.data[r] = *reinterpret_cast<const vreg_t *>(ptr);
          } else {
            vreg_t tmp(0);
            *reinterpret_cast<output_t*>(&tmp) = *ptr;
            dst.data[r] = tmp;
          }
        });
      }
    }
  }

  template <mem_layout dst_layout = row_major, typename Frag>
  static __attribute__((always_inline)) void store_matrix_sync(void *dst, const Frag &src, size_t ldm) {
    static_assert(Frag::Use == accumulator, "only accumulator fragment can be stored");
    uint32_t lane = vx_thread_id();
    if constexpr (std::is_same_v<output_t, uint16_t>) {
      uint32_t tcN_pairs = cfg::tcN / 2;
      uint32_t block_row = lane / tcN_pairs;
      uint32_t block_col = (lane % tcN_pairs) * 2;
      uint32_t m_stride  = cfg::tcM;
      uint32_t n_stride  = cfg::tcN;
      if constexpr (dst_layout == col_major) {
        std::swap(block_row, block_col);
      }
      auto base = reinterpret_cast<output_t*>(dst) + block_row * ldm + block_col;
      detail::unroll_for<Frag::NR>([&](auto r) {
        uint32_t block_m  = r / cfg::n_steps;
        uint32_t block_n  = r % cfg::n_steps;
        uint32_t elem_row = block_m * m_stride;
        uint32_t elem_col = block_n * n_stride;
        if constexpr (dst_layout == col_major) {
          std::swap(elem_row, elem_col);
        }
        auto ptr = base + elem_row * ldm + elem_col;
        auto packed = *reinterpret_cast<const uint32_t*>(&src.data[r]);
        ptr[0] = static_cast<output_t>(packed & 0xffff);
        ptr[1] = static_cast<output_t>((packed >> 16) & 0xffff);
      });
    } else {
      uint32_t block_row = lane / cfg::tcN;
      uint32_t block_col = lane % cfg::tcN;
      uint32_t m_stride  = cfg::tcM;
      uint32_t n_stride  = cfg::tcN;
      if constexpr (dst_layout == col_major) {
        std::swap(block_row, block_col);
      }
      auto base = reinterpret_cast<output_t*>(dst) + block_row * ldm + block_col;
      detail::unroll_for<Frag::NR>([&](auto r) {
        uint32_t block_m  = r / cfg::n_steps;
        uint32_t block_n  = r % cfg::n_steps;
        uint32_t elem_row = block_m * m_stride;
        uint32_t elem_col = block_n * n_stride;
        if constexpr (dst_layout == col_major) {
          std::swap(elem_row, elem_col);
        }
        auto ptr = base + elem_row * ldm + elem_col;
        if constexpr (sizeof(vreg_t) == sizeof(output_t)) {
          *reinterpret_cast<vreg_t*>(ptr) = src.data[r];
        } else {
          vreg_t tmp(src.data[r]);
          *ptr = *reinterpret_cast<const output_t*>(&tmp);
        }
      });
    }
  }

  template <typename FragD, typename FragA, typename FragB, typename FragC>
  static __attribute__((always_inline)) void mma_sync(FragD &fragD, const FragA &fragA, const FragB &fragB, const FragC &fragC) {
    static_assert(FragA::Use == matrix_a, "A must be matrix_a");
    static_assert(FragB::Use == matrix_b, "B must be matrix_b");
    static_assert(FragC::Use == accumulator, "C must be accumulator");
    static_assert(FragD::Use == accumulator, "D must be accumulator");

    // fragA: caller-saved registers (f0-f7)
    register float fa0 __asm__("f0")  = fragA.data[0];
    register float fa1 __asm__("f1")  = fragA.data[1];
    register float fa2 __asm__("f2")  = fragA.data[2];
    register float fa3 __asm__("f3")  = fragA.data[3];
    register float fa4 __asm__("f4")  = fragA.data[4];
    register float fa5 __asm__("f5")  = fragA.data[5];
    register float fa6 __asm__("f6")  = fragA.data[6];
    register float fa7 __asm__("f7")  = fragA.data[7];

    if constexpr (FragB::NR == 8) {
      // fragB: caller-saved registers (f10-f17)
      register float fb0 __asm__("f10") = fragB.data[0];
      register float fb1 __asm__("f11") = fragB.data[1];
      register float fb2 __asm__("f12") = fragB.data[2];
      register float fb3 __asm__("f13") = fragB.data[3];
      register float fb4 __asm__("f14") = fragB.data[4];
      register float fb5 __asm__("f15") = fragB.data[5];
      register float fb6 __asm__("f16") = fragB.data[6];
      register float fb7 __asm__("f17") = fragB.data[7];

      // fragC: mix of caller-saved (f28-f31) and callee-saved (f18-f21)
      float c0 = (FragC::NR > 0) ? fragC.data[0] : 0.0f;
      float c1 = (FragC::NR > 1) ? fragC.data[1] : 0.0f;
      float c2 = (FragC::NR > 2) ? fragC.data[2] : 0.0f;
      float c3 = (FragC::NR > 3) ? fragC.data[3] : 0.0f;
      float c4 = (FragC::NR > 4) ? fragC.data[4] : 0.0f;
      float c5 = (FragC::NR > 5) ? fragC.data[5] : 0.0f;
      float c6 = (FragC::NR > 6) ? fragC.data[6] : 0.0f;
      float c7 = (FragC::NR > 7) ? fragC.data[7] : 0.0f;
      register float fc0 __asm__("f24") = c0;
      register float fc1 __asm__("f25") = c1;
      register float fc2 __asm__("f26") = c2;
      register float fc3 __asm__("f27") = c3;
      register float fc4 __asm__("f28") = c4;
      register float fc5 __asm__("f29") = c5;
      register float fc6 __asm__("f30") = c6;
      register float fc7 __asm__("f31") = c7;

      // Force outputs into accumulator registers
      register float fd0 __asm__("f24");
      register float fd1 __asm__("f25");
      register float fd2 __asm__("f26");
      register float fd3 __asm__("f27");
      register float fd4 __asm__("f28");
      register float fd5 __asm__("f29");
      register float fd6 __asm__("f30");
      register float fd7 __asm__("f31");

      __asm__ volatile (".insn r %[insn], 0, 2, x%[fmd], x%[fma], x%[fmb]"
        : "=f"(fd0), "=f"(fd1), "=f"(fd2), "=f"(fd3), "=f"(fd4), "=f"(fd5), "=f"(fd6), "=f"(fd7)
        : [insn]"i"(RISCV_CUSTOM0), [fmd]"i"(Ot::id), [fma]"i"(At::id), [fmb]"i"(Bt::id),
          "f"(fa0), "f"(fa1), "f"(fa2), "f"(fa3), "f"(fa4), "f"(fa5), "f"(fa6), "f"(fa7),
          "f"(fb0), "f"(fb1), "f"(fb2), "f"(fb3), "f"(fb4), "f"(fb5), "f"(fb6), "f"(fb7),
          "f"(fc0), "f"(fc1), "f"(fc2), "f"(fc3), "f"(fc4), "f"(fc5), "f"(fc6), "f"(fc7)
      );

      // Write results to fragD
      if constexpr (FragD::NR == 8) {
        fragD.data = {fd0, fd1, fd2, fd3, fd4, fd5, fd6, fd7};
      } else {
        static_assert(FragD::NR == 4, "Unsupported accumulator register count");
        fragD.data = {fd0, fd1, fd2, fd3};
      }
    } else {
      static_assert(FragB::NR == 4, "Unsupported number of registers for FragB");
      // fragB: caller-saved registers (f28-f31)
      register float fb0 __asm__("f28") = fragB.data[0];
      register float fb1 __asm__("f29") = fragB.data[1];
      register float fb2 __asm__("f30") = fragB.data[2];
      register float fb3 __asm__("f31") = fragB.data[3];

      // fragC: mix of caller-saved (f10-f17)
      float c0 = (FragC::NR > 0) ? fragC.data[0] : 0.0f;
      float c1 = (FragC::NR > 1) ? fragC.data[1] : 0.0f;
      float c2 = (FragC::NR > 2) ? fragC.data[2] : 0.0f;
      float c3 = (FragC::NR > 3) ? fragC.data[3] : 0.0f;
      float c4 = (FragC::NR > 4) ? fragC.data[4] : 0.0f;
      float c5 = (FragC::NR > 5) ? fragC.data[5] : 0.0f;
      float c6 = (FragC::NR > 6) ? fragC.data[6] : 0.0f;
      float c7 = (FragC::NR > 7) ? fragC.data[7] : 0.0f;
      register float fc0 __asm__("f10") = c0;
      register float fc1 __asm__("f11") = c1;
      register float fc2 __asm__("f12") = c2;
      register float fc3 __asm__("f13") = c3;
      register float fc4 __asm__("f14") = c4;
      register float fc5 __asm__("f15") = c5;
      register float fc6 __asm__("f16") = c6;
      register float fc7 __asm__("f17") = c7;

      // Force outputs into accumulator registers
      register float fd0 __asm__("f10");
      register float fd1 __asm__("f11");
      register float fd2 __asm__("f12");
      register float fd3 __asm__("f13");
      register float fd4 __asm__("f14");
      register float fd5 __asm__("f15");
      register float fd6 __asm__("f16");
      register float fd7 __asm__("f17");

      __asm__ volatile (".insn r %[insn], 0, 2, x%[fmd], x%[fma], x%[fmb]"
        : "=f"(fd0), "=f"(fd1), "=f"(fd2), "=f"(fd3), "=f"(fd4), "=f"(fd5), "=f"(fd6), "=f"(fd7)
        : [insn]"i"(RISCV_CUSTOM0), [fmd]"i"(Ot::id), [fma]"i"(At::id), [fmb]"i"(Bt::id),
          "f"(fa0), "f"(fa1), "f"(fa2), "f"(fa3), "f"(fa4), "f"(fa5), "f"(fa6), "f"(fa7),
          "f"(fb0), "f"(fb1), "f"(fb2), "f"(fb3),
          "f"(fc0), "f"(fc1), "f"(fc2), "f"(fc3), "f"(fc4), "f"(fc5), "f"(fc6), "f"(fc7)
      );

      // Write results to fragD
      if constexpr (FragD::NR == 8) {
        fragD.data = {fd0, fd1, fd2, fd3, fd4, fd5, fd6, fd7};
      } else {
        static_assert(FragD::NR == 4, "Unsupported accumulator register count");
        fragD.data = {fd0, fd1, fd2, fd3};
      }
    }
  }

  template <uint32_t DescId,
            uint32_t ASlotId = 0,
            uint32_t BSlotId = ASlotId,
            uint32_t CSlotId = ASlotId,
            typename FragD,
            typename FragA,
            typename FragB,
            typename FragC>
  static __attribute__((always_inline)) void mma_sync(FragD &fragD, const FragA &fragA, const FragB &fragB, const FragC &fragC) {
    static_assert(DescId < max_static_descriptor_id, "desc_id must fit in the encoded desc_id field");
    static_assert(ASlotId < max_operand_slots, "a_slot_id out of range");
    static_assert(BSlotId < max_operand_slots, "b_slot_id out of range");
    static_assert(CSlotId < max_operand_slots, "c_slot_id out of range");
    static_assert(FragA::Use == matrix_a, "A must be matrix_a");
    static_assert(FragB::Use == matrix_b, "B must be matrix_b");
    static_assert(FragC::Use == accumulator, "C must be accumulator");
    static_assert(FragD::Use == accumulator, "D must be accumulator");

    constexpr uint32_t AbSlotCtl = encode_wmma_slot_control(ASlotId, BSlotId);

    register float fa0 __asm__("f0")  = fragA.data[0];
    register float fa1 __asm__("f1")  = fragA.data[1];
    register float fa2 __asm__("f2")  = fragA.data[2];
    register float fa3 __asm__("f3")  = fragA.data[3];
    register float fa4 __asm__("f4")  = fragA.data[4];
    register float fa5 __asm__("f5")  = fragA.data[5];
    register float fa6 __asm__("f6")  = fragA.data[6];
    register float fa7 __asm__("f7")  = fragA.data[7];

    if constexpr (FragB::NR == 8) {
      register float fb0 __asm__("f10") = fragB.data[0];
      register float fb1 __asm__("f11") = fragB.data[1];
      register float fb2 __asm__("f12") = fragB.data[2];
      register float fb3 __asm__("f13") = fragB.data[3];
      register float fb4 __asm__("f14") = fragB.data[4];
      register float fb5 __asm__("f15") = fragB.data[5];
      register float fb6 __asm__("f16") = fragB.data[6];
      register float fb7 __asm__("f17") = fragB.data[7];

      float c0 = (FragC::NR > 0) ? fragC.data[0] : 0.0f;
      float c1 = (FragC::NR > 1) ? fragC.data[1] : 0.0f;
      float c2 = (FragC::NR > 2) ? fragC.data[2] : 0.0f;
      float c3 = (FragC::NR > 3) ? fragC.data[3] : 0.0f;
      float c4 = (FragC::NR > 4) ? fragC.data[4] : 0.0f;
      float c5 = (FragC::NR > 5) ? fragC.data[5] : 0.0f;
      float c6 = (FragC::NR > 6) ? fragC.data[6] : 0.0f;
      float c7 = (FragC::NR > 7) ? fragC.data[7] : 0.0f;
      register float fc0 __asm__("f24") = c0;
      register float fc1 __asm__("f25") = c1;
      register float fc2 __asm__("f26") = c2;
      register float fc3 __asm__("f27") = c3;
      register float fc4 __asm__("f28") = c4;
      register float fc5 __asm__("f29") = c5;
      register float fc6 __asm__("f30") = c6;
      register float fc7 __asm__("f31") = c7;

      register float fd0 __asm__("f24");
      register float fd1 __asm__("f25");
      register float fd2 __asm__("f26");
      register float fd3 __asm__("f27");
      register float fd4 __asm__("f28");
      register float fd5 __asm__("f29");
      register float fd6 __asm__("f30");
      register float fd7 __asm__("f31");

      __asm__ volatile (".insn r %[insn], 0, 4, x%[desc_id], x%[ab_slot_ctl], x%[c_slot_id]"
        : "=f"(fd0), "=f"(fd1), "=f"(fd2), "=f"(fd3), "=f"(fd4), "=f"(fd5), "=f"(fd6), "=f"(fd7)
        : [insn]"i"(RISCV_CUSTOM0), [desc_id]"i"(DescId), [ab_slot_ctl]"i"(AbSlotCtl), [c_slot_id]"i"(CSlotId),
          "f"(fa0), "f"(fa1), "f"(fa2), "f"(fa3), "f"(fa4), "f"(fa5), "f"(fa6), "f"(fa7),
          "f"(fb0), "f"(fb1), "f"(fb2), "f"(fb3), "f"(fb4), "f"(fb5), "f"(fb6), "f"(fb7),
          "f"(fc0), "f"(fc1), "f"(fc2), "f"(fc3), "f"(fc4), "f"(fc5), "f"(fc6), "f"(fc7)
      );

      if constexpr (FragD::NR == 8) {
        fragD.data = {fd0, fd1, fd2, fd3, fd4, fd5, fd6, fd7};
      } else {
        static_assert(FragD::NR == 4, "Unsupported accumulator register count");
        fragD.data = {fd0, fd1, fd2, fd3};
      }
    } else {
      static_assert(FragB::NR == 4, "Unsupported number of registers for FragB");
      register float fb0 __asm__("f28") = fragB.data[0];
      register float fb1 __asm__("f29") = fragB.data[1];
      register float fb2 __asm__("f30") = fragB.data[2];
      register float fb3 __asm__("f31") = fragB.data[3];

      float c0 = (FragC::NR > 0) ? fragC.data[0] : 0.0f;
      float c1 = (FragC::NR > 1) ? fragC.data[1] : 0.0f;
      float c2 = (FragC::NR > 2) ? fragC.data[2] : 0.0f;
      float c3 = (FragC::NR > 3) ? fragC.data[3] : 0.0f;
      float c4 = (FragC::NR > 4) ? fragC.data[4] : 0.0f;
      float c5 = (FragC::NR > 5) ? fragC.data[5] : 0.0f;
      float c6 = (FragC::NR > 6) ? fragC.data[6] : 0.0f;
      float c7 = (FragC::NR > 7) ? fragC.data[7] : 0.0f;
      register float fc0 __asm__("f10") = c0;
      register float fc1 __asm__("f11") = c1;
      register float fc2 __asm__("f12") = c2;
      register float fc3 __asm__("f13") = c3;
      register float fc4 __asm__("f14") = c4;
      register float fc5 __asm__("f15") = c5;
      register float fc6 __asm__("f16") = c6;
      register float fc7 __asm__("f17") = c7;

      register float fd0 __asm__("f10");
      register float fd1 __asm__("f11");
      register float fd2 __asm__("f12");
      register float fd3 __asm__("f13");
      register float fd4 __asm__("f14");
      register float fd5 __asm__("f15");
      register float fd6 __asm__("f16");
      register float fd7 __asm__("f17");

      __asm__ volatile (".insn r %[insn], 0, 4, x%[desc_id], x%[ab_slot_ctl], x%[c_slot_id]"
        : "=f"(fd0), "=f"(fd1), "=f"(fd2), "=f"(fd3), "=f"(fd4), "=f"(fd5), "=f"(fd6), "=f"(fd7)
        : [insn]"i"(RISCV_CUSTOM0), [desc_id]"i"(DescId), [ab_slot_ctl]"i"(AbSlotCtl), [c_slot_id]"i"(CSlotId),
          "f"(fa0), "f"(fa1), "f"(fa2), "f"(fa3), "f"(fa4), "f"(fa5), "f"(fa6), "f"(fa7),
          "f"(fb0), "f"(fb1), "f"(fb2), "f"(fb3),
          "f"(fc0), "f"(fc1), "f"(fc2), "f"(fc3), "f"(fc4), "f"(fc5), "f"(fc6), "f"(fc7)
      );

      if constexpr (FragD::NR == 8) {
        fragD.data = {fd0, fd1, fd2, fd3, fd4, fd5, fd6, fd7};
      } else {
        static_assert(FragD::NR == 4, "Unsupported accumulator register count");
        fragD.data = {fd0, fd1, fd2, fd3};
      }
    }
  }

  template <uint32_t DescId, typename FragD, typename FragA, typename FragB, typename FragC>
  static __attribute__((always_inline)) void mma_sync_slots(uint32_t a_slot_id,
                                                            uint32_t b_slot_id,
                                                            uint32_t c_slot_id,
                                                            FragD &fragD,
                                                            const FragA &fragA,
                                                            const FragB &fragB,
                                                            const FragC &fragC) {
    switch ((a_slot_id << 2) | (b_slot_id << 1) | c_slot_id) {
    case 0: mma_sync<DescId, 0, 0, 0>(fragD, fragA, fragB, fragC); break;
    case 1: mma_sync<DescId, 0, 0, 1>(fragD, fragA, fragB, fragC); break;
    case 2: mma_sync<DescId, 0, 1, 0>(fragD, fragA, fragB, fragC); break;
    case 3: mma_sync<DescId, 0, 1, 1>(fragD, fragA, fragB, fragC); break;
    case 4: mma_sync<DescId, 1, 0, 0>(fragD, fragA, fragB, fragC); break;
    case 5: mma_sync<DescId, 1, 0, 1>(fragD, fragA, fragB, fragC); break;
    case 6: mma_sync<DescId, 1, 1, 0>(fragD, fragA, fragB, fragC); break;
    case 7: mma_sync<DescId, 1, 1, 1>(fragD, fragA, fragB, fragC); break;
    default: __builtin_trap();
    }
  }

  template <uint32_t DescId, typename FragD, typename FragA, typename FragB, typename FragC>
  static __attribute__((always_inline)) void mma_sync_slot(uint32_t slot_id,
                                                           FragD &fragD,
                                                           const FragA &fragA,
                                                           const FragB &fragB,
                                                           const FragC &fragC) {
    switch (slot_id) {
    case 0: mma_sync<DescId, 0, 0, 0>(fragD, fragA, fragB, fragC); break;
    case 1: mma_sync<DescId, 1, 1, 1>(fragD, fragA, fragB, fragC); break;
    default: __builtin_trap();
    }
  }
};

template <uint32_t NT,
          typename It,
          typename Ot>
using wmma_context = wmma_context_ab<NT, It, It, Ot>;

} // namespace tensor
} // namespace vortex
