#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context_ab<NUM_THREADS, vt::ATYPE, vt::BTYPE, vt::OTYPE>;

static constexpr uint32_t kMGroups    = M_GROUPS;
static constexpr uint32_t kNGroups    = N_GROUPS;
static constexpr uint32_t kKPhases    = K_PHASES;
static constexpr uint32_t kKTilesWin  = K_TILES_PER_WIN;
static constexpr uint32_t kATileCols  = A_TILE_COLS;
static constexpr uint32_t kBTileCols  = B_TILE_COLS;
static constexpr uint32_t kCTileCols  = C_TILE_COLS;
static constexpr uint32_t kMmaDescId  = 0;

// Descriptor layout
static constexpr uint32_t kACount = kMGroups * kKPhases;
static constexpr uint32_t kBBase  = kACount;
static constexpr uint32_t kCInId  = kACount + kKPhases * kNGroups;
static constexpr uint32_t kDBase  = kCInId + 1;

static inline uint32_t a_desc(uint32_t mg, uint32_t k) { return mg * kKPhases + k; }
static inline uint32_t b_desc(uint32_t k, uint32_t ng) { return kBBase + k * kNGroups + ng; }
static inline uint32_t d_desc(uint32_t mg, uint32_t ng) { return kDBase + mg * kNGroups + ng; }

static inline uint32_t a_tile(uint32_t m_blk, uint32_t k_step) {
  return m_blk * kATileCols + k_step;
}
static inline uint32_t b_tile(uint32_t k_step, uint32_t n_blk) {
  return k_step * kBTileCols + n_blk;
}
static inline uint32_t c_tile(uint32_t m_blk, uint32_t n_blk) {
  return m_blk * kCTileCols + n_blk;
}

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
// kernel_body — 双缓冲流水线 (2 allocations)
//
// 两个 TMEM allocation 交替使用: handle[0] 和 handle[1]
// 当 compute 读取 handle[cur] 的 A/B window 时,
// TMA 同时向 handle[nxt] 搬运下一个 K-phase 的 A/B 数据。
//
// TMA 写入 handle[nxt] (TMEM 列 [32..63]) 和
// mma_load 读取 handle[cur] (TMEM 列 [0..31]) 不冲突 → 可并行。
//
// CMem 不属于 TMEM allocation, 是全局共享的。
// 输出驻留模式下 FIFO 持续累加, 不受 handle 切换影响。
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

  // 2 个 allocation, 各占 32 lines (fp8 all, col=64)
  uint32_t handle[2];
  handle[0] = vt::tmem_alloc(arg->col_span, kMmaDescId);
  handle[1] = vt::tmem_alloc(arg->col_span, kMmaDescId);

  for (uint32_t ng = 0; ng < kNGroups; ++ng) {

    for (uint32_t batch = 0; batch < OUTPUT_BATCHES; ++batch) {
      uint32_t m_blk = batch;

      // 加载 C = 0 (用 handle[0] 的 C window)
      (void)vt::tma_load(handle[0], kCInId);
      for (uint32_t n_blk = 0; n_blk < TILES_PER_BATCH; ++n_blk) {
        vt::mbarrier_init(kBarC, 1);
        vt::mma_load_c_slot<kMmaDescId>(handle[0], c_tile(m_blk, n_blk), n_blk);
        (void)vt::tc_commit(kBarC);
        vt::mbarrier_arrive(kBarC);
        vt::mbarrier_wait(kBarC);
      }

      // 预加载第一个 K-phase 到 handle[0]
      (void)vt::tma_load(handle[0], a_desc(m_group, 0));
      (void)vt::tma_load(handle[0], b_desc(0, ng));

      // ---- K 循环: 双缓冲流水线 ----
      for (uint32_t kp = 0; kp < kKPhases; ++kp) {
        uint32_t cur = kp & 1;
        uint32_t nxt = 1 - cur;

        // 预取下一 K-phase 到另一个 handle (与当前 compute 重叠)
        if (kp + 1 < kKPhases) {
          (void)vt::tma_load(handle[nxt], a_desc(m_group, kp + 1));
          (void)vt::tma_load(handle[nxt], b_desc(kp + 1, ng));
        }

        // 在 handle[cur] 上计算: 2 output tiles × 4 k8 steps
        // 先处理完一个 output tile 的所有 k8 再切换 (减少 FIFO switch)
        for (uint32_t n_blk = 0; n_blk < TILES_PER_BATCH; ++n_blk) {
          uint32_t c_slot = n_blk;
          for (uint32_t ks = 0; ks < kKTilesWin; ++ks) {
            uint32_t ab_slot = ks & 1;
            load_ab(handle[cur], a_tile(m_blk, ks), b_tile(ks, n_blk), ab_slot);
            vt::mbarrier_wait(kBarAB + ab_slot);
            ctx::mma_sync_slots<kMmaDescId>(ab_slot, ab_slot, c_slot,
                                             fragC, fragA, fragB, fragC);
          }
        }
      }

      // 存储 (用最后一个 K-phase 的 handle)
      uint32_t last_h = (kKPhases - 1) & 1;
      vt::tc_wait();
      for (uint32_t n_blk = 0; n_blk < TILES_PER_BATCH; ++n_blk) {
        vt::mma_store_c_slot<kMmaDescId>(handle[last_h], c_tile(m_blk, n_blk), n_blk);
      }
      vt::tc_wait();
      (void)vt::tma_store(handle[last_h], d_desc(m_group, ng));
    }
  }

  vt::tmem_free(handle[1]);
  vt::tmem_free(handle[0]);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  uint32_t grid_dim = kMGroups;
  return vx_spawn_threads(1, &grid_dim, arg->block_dim,
                          (vx_kernel_func_cb)kernel_body, arg);
}
