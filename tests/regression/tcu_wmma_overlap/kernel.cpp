#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context_ab<NUM_THREADS, vt::ATYPE, vt::BTYPE, vt::OTYPE>;

static constexpr uint32_t kMTiles   = M_TILES;
static constexpr uint32_t kNTiles   = N_TILES;
static constexpr uint32_t kKPhases  = K_PHASES;
static constexpr uint32_t kATilesPerWin = A_TILES_PER_WIN;  // 2
static constexpr uint32_t kMmaDescId = 0;

// ---------------------------------------------------------------------------
// Descriptor-table layout (shared with host):
//   A descs : [0 .. kMTiles*kKPhases)              per (m, k_phase)
//   B descs : [kBBase .. kBBase+kKPhases*kNTiles)   per (k_phase, n)
//   C_in    : kCInId                                single zero-buffer
//   D_out   : [kDBase .. kDBase+kMTiles*kNTiles)    per (m, n)
// ---------------------------------------------------------------------------
static constexpr uint32_t kACount  = kMTiles * kKPhases;
static constexpr uint32_t kBBase   = kACount;
static constexpr uint32_t kCInId   = kACount + kKPhases * kNTiles;
static constexpr uint32_t kDBase   = kCInId + 1;

static inline uint32_t a_desc(uint32_t m, uint32_t k) { return m * kKPhases + k; }
static inline uint32_t b_desc(uint32_t k, uint32_t n) { return kBBase + k * kNTiles + n; }
static inline uint32_t d_desc(uint32_t m, uint32_t n) { return kDBase + m * kNTiles + n; }

// Barrier IDs
static constexpr uint32_t kBarrier0 = 0;
static constexpr uint32_t kBarrier1 = 1;
static constexpr uint32_t kCBarrier = 2;

static inline void issue_mma_load_ab(uint32_t handle,
                                     uint32_t a_tile, uint32_t b_tile,
                                     uint32_t slot, uint32_t barrier) {
  vt::mbarrier_init(barrier, 1);
  vt::mma_load_a_slot<kMmaDescId>(handle, a_tile, slot);
  vt::mma_load_b_slot<kMmaDescId>(handle, b_tile, slot);
  (void)vt::tc_commit(barrier);
  vt::mbarrier_arrive(barrier);
}

// ---------------------------------------------------------------------------
// kernel_body — 32 warps (warpgroup), each processes one M-row of N-tiles.
//
// Window shape: 16×16 (m16n16k16)
//   A window: 16×16 → 2 tiles (k8_step 0,1)
//   B window: 16×16 → 2 tiles (k8_step 0,1)
//   C/D window: 16×16 → 1 tile
//
// Output-resident (ws=1): FIFO accumulates across all K-phases.
// AMem/BMem ping-pong: slot 0/1 alternates within each k16 window.
// CMem ping-pong: slot 0/1 alternates between consecutive N-tiles.
// ---------------------------------------------------------------------------
static inline void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t m_tile = blockIdx.x;
  if (m_tile >= kMTiles) return;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragA, 0);
  ctx::fill_fragment(fragB, 0);
  ctx::fill_fragment(fragC, 0);

  // 一个 allocation: A+B+C+D windows, col_span 由 host 传入
  uint32_t handle = vt::tmem_alloc(arg->col_span, kMmaDescId);

  uint32_t d_store_async[2] = {};
  bool     d_store_live[2]  = {false, false};

  // ---- 外层 N-tile 循环 (CMem ping-pong) ----
  for (uint32_t n_tile = 0; n_tile < kNTiles; ++n_tile) {
    uint32_t c_slot = n_tile & 1;

    // 等待同一 c_slot 的上一次 TMA store 完成
    if (d_store_live[c_slot]) {
      vt::tma_wait(d_store_async[c_slot]);
      d_store_live[c_slot] = false;
    }

    // 加载 C = 0: DRAM → TMEM(C window) → CMem[c_slot]
    (void)vt::tma_load(handle, kCInId);
    vt::mbarrier_init(kCBarrier, 1);
    vt::mma_load_c_slot<kMmaDescId>(handle, 0, c_slot);
    (void)vt::tc_commit(kCBarrier);
    vt::mbarrier_arrive(kCBarrier);
    vt::mbarrier_wait(kCBarrier);

    // ---- 内层 K-phase 循环 (A/B ping-pong, output-resident 累加) ----
    for (uint32_t k = 0; k < kKPhases; ++k) {
      // TMA 搬运一个 16×16 A/B window (含 2 个 k8 tile)
      (void)vt::tma_load(handle, a_desc(m_tile, k));
      (void)vt::tma_load(handle, b_desc(k, n_tile));

      // 遍历 window 内的 2 个 k8 tile, AMem/BMem slot ping-pong
      for (uint32_t t = 0; t < kATilesPerWin; ++t) {
        uint32_t ab_slot = t & 1;
        issue_mma_load_ab(handle, t, t, ab_slot, ab_slot);
        vt::mbarrier_wait(ab_slot);

        // WMMA: CMem[c_slot] += AMem[ab_slot] * BMem[ab_slot]
        ctx::mma_sync_slots<kMmaDescId>(ab_slot, ab_slot, c_slot,
                                         fragC, fragA, fragB, fragC);
      }
    }

    // 存储累加结果: CMem → TMEM(D window) → DRAM
    vt::tc_wait();
    vt::mma_store_c_slot<kMmaDescId>(handle, 0, c_slot);
    vt::tc_wait();
    d_store_async[c_slot] = vt::tma_store(handle, d_desc(m_tile, n_tile));
    d_store_live[c_slot] = true;
  }

  // 排空未完成的 store
  for (uint32_t s = 0; s < 2; ++s) {
    if (d_store_live[s])
      vt::tma_wait(d_store_async[s]);
  }

  vt::tmem_free(handle);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  uint32_t grid_dim = kMTiles;  // 32 warps = warpgroup
  return vx_spawn_threads(1, &grid_dim, arg->block_dim,
                          (vx_kernel_func_cb)kernel_body, arg);
}
