#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context_ab<NUM_THREADS, vt::ATYPE, vt::BTYPE, vt::OTYPE>;

static constexpr uint32_t kMmaDescId = 0;
static constexpr uint32_t kPipelineStages = 2;
static constexpr uint32_t kWorkerWarps = WORKER_WARPS;

static inline void free_worker_handles(uint32_t handle_a[kPipelineStages],
                                       uint32_t handle_b[kPipelineStages],
                                       uint32_t handle_c,
                                       uint32_t handle_d) {
  vt::tmem_free(handle_d);
  vt::tmem_free(handle_c);
  for (uint32_t stage = 0; stage < kPipelineStages; ++stage) {
    vt::tmem_free(handle_b[stage]);
    vt::tmem_free(handle_a[stage]);
  }
}

static inline uint32_t a_desc_base(const kernel_arg_t* arg) {
  (void)arg;
  return 0;
}

static inline uint32_t b_desc_base(const kernel_arg_t* arg) {
  uint32_t num_tiles = arg->tile_grid[0] * arg->tile_grid[1];
  return arg->phase_limit * num_tiles;
}

static inline uint32_t c_in_desc_base(const kernel_arg_t* arg) {
  uint32_t num_tiles = arg->tile_grid[0] * arg->tile_grid[1];
  return 2 * arg->phase_limit * num_tiles;
}

static inline uint32_t c_out_desc_base(const kernel_arg_t* arg) {
  uint32_t num_tiles = arg->tile_grid[0] * arg->tile_grid[1];
  return 2 * arg->phase_limit * num_tiles + num_tiles;
}

static inline uint32_t phase_desc_id(uint32_t base, uint32_t phase, uint32_t num_tiles, uint32_t tile_id) {
  return base + phase * num_tiles + tile_id;
}

static inline void prepare_c_tile(kernel_arg_t* __UNIFORM__ arg,
                                  uint32_t tile_id,
                                  uint32_t handle_c,
                                  uint32_t slot_id) {
  (void)vt::tma_load(handle_c, c_in_desc_base(arg) + tile_id);
  vt::mma_load_c_slot<kMmaDescId>(handle_c, 0, 0, slot_id);
}

static inline void prepare_ab_phase(kernel_arg_t* __UNIFORM__ arg,
                                    uint32_t tile_id,
                                    uint32_t phase,
                                    uint32_t stage,
                                    uint32_t handle_a[kPipelineStages],
                                    uint32_t handle_b[kPipelineStages],
                                    uint32_t num_tiles) {
  (void)vt::tma_load(handle_a[stage],
                     phase_desc_id(a_desc_base(arg), phase, num_tiles, tile_id));
  (void)vt::tma_load(handle_b[stage],
                     phase_desc_id(b_desc_base(arg), phase, num_tiles, tile_id));
  vt::mma_load_a_slot<kMmaDescId>(handle_a[stage], 0, 0, stage);
  vt::mma_load_b_slot<kMmaDescId>(handle_b[stage], 0, 0, stage);
}

static inline void run_tile(kernel_arg_t* __UNIFORM__ arg,
                            uint32_t tile_id,
                            uint32_t handle_a[kPipelineStages],
                            uint32_t handle_b[kPipelineStages],
                            uint32_t handle_c,
                            uint32_t handle_d,
                            bool c_prefetched,
                            bool phase0_prefetched,
                            bool phase1_prefetched,
                            uint32_t c_slot,
                            bool prefetch_next_tile,
                            bool* next_c_prefetched,
                            bool* next_phase0_prefetched,
                            bool* next_phase1_prefetched,
                            uint32_t* pending_store_id,
                            bool* pending_store_valid) {
  ctx::fragment_a fragA;
  ctx::fragment_b fragB;
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragA, 0);
  ctx::fill_fragment(fragB, 0);
  ctx::fill_fragment(fragC, 0);

  uint32_t num_tiles = arg->tile_grid[0] * arg->tile_grid[1];
  if (!c_prefetched) {
    prepare_c_tile(arg, tile_id, handle_c, c_slot);
  }

  if (arg->phase_limit != 0) {
    if (!phase0_prefetched) {
      prepare_ab_phase(arg, tile_id, 0, 0, handle_a, handle_b, num_tiles);
    }
  }
  if (arg->phase_limit > 1) {
    if (!phase1_prefetched) {
      prepare_ab_phase(arg, tile_id, 1, 1, handle_a, handle_b, num_tiles);
    }
  }

  for (uint32_t phase_base = 0; phase_base < arg->phase_limit; phase_base += kPipelineStages) {
    if (phase_base < arg->phase_limit) {
      ctx::mma_sync_slots<kMmaDescId>(0, 0, c_slot, fragC, fragA, fragB, fragC);
    }
    if (phase_base + 1 < arg->phase_limit) {
      ctx::mma_sync_slots<kMmaDescId>(1, 1, c_slot, fragC, fragA, fragB, fragC);
    }
    if (phase_base + 2 < arg->phase_limit) {
      prepare_ab_phase(arg, tile_id, phase_base + 2, 0, handle_a, handle_b, num_tiles);
    }
    if (phase_base + 3 < arg->phase_limit) {
      prepare_ab_phase(arg, tile_id, phase_base + 3, 1, handle_a, handle_b, num_tiles);
    }
  }

  vt::tc_wait();
  if (prefetch_next_tile) {
    uint32_t next_tile_id = tile_id + 1;
    uint32_t next_c_slot = c_slot ^ 1u;
    prepare_c_tile(arg, next_tile_id, handle_c, next_c_slot);
    if (next_c_prefetched) {
      *next_c_prefetched = true;
    }
    if (arg->phase_limit != 0) {
      prepare_ab_phase(arg, next_tile_id, 0, 0, handle_a, handle_b, num_tiles);
      if (next_phase0_prefetched) {
        *next_phase0_prefetched = true;
      }
    }
    if (arg->phase_limit > 1) {
      prepare_ab_phase(arg, next_tile_id, 1, 1, handle_a, handle_b, num_tiles);
      if (next_phase1_prefetched) {
        *next_phase1_prefetched = true;
      }
    }
  } else {
    if (next_c_prefetched) {
      *next_c_prefetched = false;
    }
    if (next_phase0_prefetched) {
      *next_phase0_prefetched = false;
    }
    if (next_phase1_prefetched) {
      *next_phase1_prefetched = false;
    }
  }

  if (pending_store_valid && *pending_store_valid) {
    vt::tma_wait(*pending_store_id);
    *pending_store_valid = false;
  }
  vt::mma_store_c_slot<kMmaDescId>(handle_d, 0, 0, c_slot);
  vt::tc_wait();
  *pending_store_id = vt::tma_store(handle_d, c_out_desc_base(arg) + tile_id);
  *pending_store_valid = true;
}

static inline void run_worker_single(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t num_tiles = arg->tile_grid[0] * arg->tile_grid[1];
  if (0 == num_tiles) {
    return;
  }

  uint32_t handle_a[kPipelineStages];
  uint32_t handle_b[kPipelineStages];
  for (uint32_t stage = 0; stage < kPipelineStages; ++stage) {
    handle_a[stage] = vt::tmem_alloc(arg->a_bank_span);
    handle_b[stage] = vt::tmem_alloc(arg->b_bank_span);
  }
  uint32_t handle_c = vt::tmem_alloc(arg->c_bank_span);
  uint32_t handle_d = vt::tmem_alloc(arg->c_bank_span);

  bool c_prefetched = true;
  bool phase0_prefetched = (arg->phase_limit != 0);
  bool phase1_prefetched = (arg->phase_limit > 1);
  uint32_t c_slot = 0;
  uint32_t pending_store_id = 0;
  bool pending_store_valid = false;

  prepare_c_tile(arg, 0, handle_c, c_slot);
  if (phase0_prefetched) {
    prepare_ab_phase(arg, 0, 0, 0, handle_a, handle_b, num_tiles);
  }
  if (phase1_prefetched) {
    prepare_ab_phase(arg, 0, 1, 1, handle_a, handle_b, num_tiles);
  }

  for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
    bool next_c_prefetched = false;
    bool next_phase0_prefetched = false;
    bool next_phase1_prefetched = false;
    run_tile(arg,
             tile_id,
             handle_a,
             handle_b,
             handle_c,
             handle_d,
             c_prefetched,
             phase0_prefetched,
             phase1_prefetched,
             c_slot,
             tile_id + 1 < num_tiles,
             &next_c_prefetched,
             &next_phase0_prefetched,
             &next_phase1_prefetched,
             &pending_store_id,
             &pending_store_valid);
    c_prefetched = next_c_prefetched;
    phase0_prefetched = next_phase0_prefetched;
    phase1_prefetched = next_phase1_prefetched;
    c_slot ^= 1u;
  }

  if (pending_store_valid) {
    vt::tma_wait(pending_store_id);
  }

  free_worker_handles(handle_a, handle_b, handle_c, handle_d);
}

static inline void run_worker_dual(kernel_arg_t* __UNIFORM__ arg, uint32_t wid) {
  uint32_t num_tiles = arg->tile_grid[0] * arg->tile_grid[1];
  if (0 == num_tiles || wid >= kWorkerWarps) {
    return;
  }

  uint32_t slot_id = wid & 0x1;
  uint32_t handle_a = vt::tmem_alloc(arg->a_bank_span);
  uint32_t handle_b = vt::tmem_alloc(arg->b_bank_span);
  uint32_t handle_c = vt::tmem_alloc(arg->c_bank_span);
  uint32_t handle_d = vt::tmem_alloc(arg->c_bank_span);

  ctx::fragment_a fragA;
  ctx::fragment_b fragB;
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragA, 0);
  ctx::fill_fragment(fragB, 0);
  ctx::fill_fragment(fragC, 0);

  for (uint32_t tile_id = wid; tile_id < num_tiles; tile_id += kWorkerWarps) {
    prepare_c_tile(arg, tile_id, handle_c, slot_id);
    for (uint32_t phase = 0; phase < arg->phase_limit; ++phase) {
      (void)vt::tma_load(handle_a, phase_desc_id(a_desc_base(arg), phase, num_tiles, tile_id));
      (void)vt::tma_load(handle_b, phase_desc_id(b_desc_base(arg), phase, num_tiles, tile_id));
      vt::mma_load_a_slot<kMmaDescId>(handle_a, 0, 0, slot_id);
      vt::mma_load_b_slot<kMmaDescId>(handle_b, 0, 0, slot_id);
      ctx::mma_sync_slots<kMmaDescId>(slot_id, slot_id, slot_id, fragC, fragA, fragB, fragC);
    }
    vt::tc_wait();
    vt::mma_store_c_slot<kMmaDescId>(handle_d, 0, 0, slot_id);
    vt::tc_wait();
    auto store_id = vt::tma_store(handle_d, c_out_desc_base(arg) + tile_id);
    vt::tma_wait(store_id);
  }

  vt::tmem_free(handle_d);
  vt::tmem_free(handle_c);
  vt::tmem_free(handle_b);
  vt::tmem_free(handle_a);
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  if constexpr (kWorkerWarps == 1) {
    if (blockIdx.x != 0) {
      return;
    }
    run_worker_single(arg);
  } else {
    if (blockIdx.x >= kWorkerWarps) {
      return;
    }
    run_worker_dual(arg, blockIdx.x);
  }
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  uint32_t worker_warps = kWorkerWarps;
  return vx_spawn_threads(1, &worker_warps, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
