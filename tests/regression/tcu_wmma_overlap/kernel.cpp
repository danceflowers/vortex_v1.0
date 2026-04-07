#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context_ab<NUM_THREADS, vt::ATYPE, vt::BTYPE, vt::OTYPE>;

static constexpr uint32_t kMGroups    = M_GROUPS;       // 16
static constexpr uint32_t kNGroups    = N_GROUPS;       // 16
static constexpr uint32_t kKPhases    = K_PHASES;       // 16
static constexpr uint32_t kKTilesWin  = K_TILES_PER_WIN; // 4 k8 tiles per window
static constexpr uint32_t kATileCols  = A_TILE_COLS;    // 4
static constexpr uint32_t kBTileCols  = B_TILE_COLS;    // 2
static constexpr uint32_t kCTileCols  = C_TILE_COLS;    // 2
static constexpr uint32_t kMmaDescId  = 0;

// ---------------------------------------------------------------------------
// Descriptor-table layout (shared with host):
//   A descs : [0 .. kMGroups*kKPhases)              per (m_group, k_phase)
//   B descs : [kBBase .. kBBase+kKPhases*kNGroups)   per (k_phase, n_group)
//   C_in    : kCInId                                single zero-buffer
//   D_out   : [kDBase .. kDBase+kMGroups*kNGroups)   per (m_group, n_group)
// ---------------------------------------------------------------------------
static constexpr uint32_t kACount = kMGroups * kKPhases;
static constexpr uint32_t kBBase  = kACount;
static constexpr uint32_t kCInId  = kACount + kKPhases * kNGroups;
static constexpr uint32_t kDBase  = kCInId + 1;

static inline uint32_t a_desc(uint32_t mg, uint32_t k) { return mg * kKPhases + k; }
static inline uint32_t b_desc(uint32_t k, uint32_t ng) { return kBBase + k * kNGroups + ng; }
static inline uint32_t d_desc(uint32_t mg, uint32_t ng) { return kDBase + mg * kNGroups + ng; }

// A tile_id(m_block, k_step) within 32×32 A window (tile_rows=2, tile_cols=4)
static inline uint32_t a_tile(uint32_t m_blk, uint32_t k_step) {
  return m_blk * kATileCols + k_step;
}
// B tile_id(k_step, n_block) within 32×32 B window (tile_rows=4, tile_cols=2)
static inline uint32_t b_tile(uint32_t k_step, uint32_t n_blk) {
  return k_step * kBTileCols + n_blk;
}
// C/D tile_id(m_block, n_block) within 32×32 C window (tile_rows=2, tile_cols=2)
static inline uint32_t c_tile(uint32_t m_blk, uint32_t n_blk) {
  return m_blk * kCTileCols + n_blk;
}

// Barrier IDs
static constexpr uint32_t kBarAB = 0;
static constexpr uint32_t kBarC  = 2;

static inline void load_ab(uint32_t handle, uint32_t a_tid, uint32_t b_tid,
                            uint32_t slot) {
  vt::mbarrier_init(kBarAB + slot, 1);
  vt::mma_load_a_slot<kMmaDescId>(handle, a_tid, slot);
  vt::mma_load_b_slot<kMmaDescId>(handle, b_tid, slot);
  (void)vt::tc_commit(kBarAB + slot);
  vt::mbarrier_arrive(kBarAB + slot);
}

// ---------------------------------------------------------------------------
// kernel_body — 16 warps (warpgroup), each processes one M-group (32 rows).
//
// Window shape: 32×32 (m32n32k32)
//   A window 32×32 → 8 tiles (2M × 4K)
//   B window 32×32 → 8 tiles (4K × 2N)
//   C/D window 32×32 → 4 tiles (2M × 2N)
//
// 4 output tiles per 32×32 block, CMem 2 slots → 2 批次:
//   批次 0: output tiles (m=0,n=0) c_slot=0, (m=0,n=1) c_slot=1
//   批次 1: output tiles (m=1,n=0) c_slot=0, (m=1,n=1) c_slot=1
// 每批处理整个 K 循环。A/B 需加载 2 次但每次搬 1024B (TMA 高效)。
// ---------------------------------------------------------------------------
static inline void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t m_group = blockIdx.x;
  if (m_group >= kMGroups) return;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragA, 0);
  ctx::fill_fragment(fragB, 0);
  ctx::fill_fragment(fragC, 0);

  uint32_t handle = vt::tmem_alloc(arg->col_span, kMmaDescId);

  // ---- 外层 N-group 循环 ----
  for (uint32_t ng = 0; ng < kNGroups; ++ng) {

    // 2 批次处理 4 个 output tiles
    for (uint32_t batch = 0; batch < OUTPUT_BATCHES; ++batch) {
      uint32_t m_blk = batch;  // 批次 0: m_blk=0, 批次 1: m_blk=1

      // 加载 C = 0 到 2 个 c_slot
      (void)vt::tma_load(handle, kCInId);
      for (uint32_t n_blk = 0; n_blk < TILES_PER_BATCH; ++n_blk) {
        uint32_t c_slot = n_blk;
        vt::mbarrier_init(kBarC, 1);
        vt::mma_load_c_slot<kMmaDescId>(handle, c_tile(m_blk, n_blk), c_slot);
        (void)vt::tc_commit(kBarC);
        vt::mbarrier_arrive(kBarC);
        vt::mbarrier_wait(kBarC);
      }

      // ---- K 循环: 16 K-phases, 每 phase 4 k8 steps ----
      for (uint32_t kp = 0; kp < kKPhases; ++kp) {
        (void)vt::tma_load(handle, a_desc(m_group, kp));
        (void)vt::tma_load(handle, b_desc(kp, ng));

        for (uint32_t ks = 0; ks < kKTilesWin; ++ks) {
          uint32_t ab_slot = ks & 1;

          // 对 batch 内的 2 个 output tile 都做 WMMA
          for (uint32_t n_blk = 0; n_blk < TILES_PER_BATCH; ++n_blk) {
            uint32_t c_slot = n_blk;
            load_ab(handle, a_tile(m_blk, ks), b_tile(ks, n_blk), ab_slot);
            vt::mbarrier_wait(kBarAB + ab_slot);
            ctx::mma_sync_slots<kMmaDescId>(ab_slot, ab_slot, c_slot,
                                             fragC, fragA, fragB, fragC);
          }
        }
      }

      // 存储本批次的 2 个 output tile
      vt::tc_wait();
      for (uint32_t n_blk = 0; n_blk < TILES_PER_BATCH; ++n_blk) {
        vt::mma_store_c_slot<kMmaDescId>(handle, c_tile(m_blk, n_blk), n_blk);
      }
      vt::tc_wait();
      (void)vt::tma_store(handle, d_desc(m_group, ng));
    }
  }

  vt::tmem_free(handle);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  uint32_t grid_dim = kMGroups;  // 16 warps
  return vx_spawn_threads(1, &grid_dim, arg->block_dim,
                          (vx_kernel_func_cb)kernel_body, arg);
}
