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
// 两个 TMEM allocation 交替使用: handle 和 handle
// 当 compute 读取 handle 的 A/B window 时,
// TMA 同时向 handle 搬运下一个 K-phase 的 A/B 数据。
//
// TMA 写入 handle (TMEM 列 [32..63]) 和
// mma_load 读取 handle (TMEM 列 [0..31]) 不冲突 → 可并行。
//
// CMem 不属于 TMEM allocation, 是全局共享的。
// 输出驻留模式下 FIFO 持续累加, 不受 handle 切换影响。
// ---------------------------------------------------------------------------
static inline void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  // 单 warp 处理所有 M-groups (TMEM 是核心共享资源)
  for (uint32_t m_group = 0; m_group < kMGroups; ++m_group) {

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragA, 0);
  ctx::fill_fragment(fragB, 0);
  ctx::fill_fragment(fragC, 0);

  // 单 allocation: 全部 A/B/C/D windows
  // TMEM 是核心共享资源, 多 warp 串行使用
  uint32_t handle = vt::tmem_alloc(arg->col_span, kMmaDescId);

  for (uint32_t ng = 0; ng < kNGroups; ++ng) {

    for (uint32_t batch = 0; batch < OUTPUT_BATCHES; ++batch) {
      uint32_t m_blk = batch;

      // 加载 C = 0 → TMEM C window (window_id=2)
      (void)vt::tma_load(handle, kCInId, /*window_id=*/2);
      for (uint32_t n_blk = 0; n_blk < TILES_PER_BATCH; ++n_blk) {
        vt::mbarrier_init(kBarC, 1);
        vt::mma_load_c_slot<kMmaDescId>(handle, c_tile(m_blk, n_blk), n_blk);
        (void)vt::tc_commit(kBarC);
        vt::mbarrier_arrive(kBarC);
        vt::mbarrier_wait(kBarC);
      }

      // 预加载第一个 K-phase: A→window 0, B→window 1
      (void)vt::tma_load(handle, a_desc(m_group, 0), /*window_id=*/0);
      (void)vt::tma_load(handle, b_desc(0, ng), /*window_id=*/1);

      // ---- K 循环 ----
      for (uint32_t kp = 0; kp < kKPhases; ++kp) {
        if (kp + 1 < kKPhases) {
          (void)vt::tma_load(handle, a_desc(m_group, kp + 1), /*window_id=*/0);
          (void)vt::tma_load(handle, b_desc(kp + 1, ng), /*window_id=*/1);
        }

        // 计算: 2 output tiles × 4 k8 steps
        // 先处理完一个 output tile 的所有 k8 再切换 (减少 FIFO switch)
        for (uint32_t n_blk = 0; n_blk < TILES_PER_BATCH; ++n_blk) {
          uint32_t c_slot = n_blk;
          for (uint32_t ks = 0; ks < kKTilesWin; ++ks) {
            uint32_t ab_slot = ks & 1;
            load_ab(handle, a_tile(m_blk, ks), b_tile(ks, n_blk), ab_slot);
            vt::mbarrier_wait(kBarAB + ab_slot);
            ctx::mma_sync_slots<kMmaDescId>(ab_slot, ab_slot, c_slot,
                                             fragC, fragA, fragB, fragC);
          }
        }
      }

      // 存储
      vt::tc_wait();
      for (uint32_t n_blk = 0; n_blk < TILES_PER_BATCH; ++n_blk) {
        vt::mma_store_c_slot<kMmaDescId>(handle, c_tile(m_blk, n_blk), n_blk);
      }
      vt::tc_wait();
      (void)vt::tma_store(handle, d_desc(m_group, ng), /*window_id=*/3);
    }
  }

  vt::tmem_free(handle);
  } // m_group loop
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  uint32_t grid_dim = 1;  // 单 warp: TMEM 是核心共享, 不能多 warp 同时占用
  return vx_spawn_threads(1, &grid_dim, arg->block_dim,
                          (vx_kernel_func_cb)kernel_body, arg);
}
