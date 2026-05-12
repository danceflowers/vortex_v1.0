#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;

// Phase-2 migration. Legacy fragment-based mma_sync_slots and the
// tc_commit_scope::mma_load_only pipelining bypass are gone in the new ISA.
// TCU_MMA is single-instruction and there is no public scope-restricted
// commit (mbar_commit registers all inflight async tcgen05 ops). The
// double-buffer pipeline is preserved at the cpabulk + TCU_MMA level; data
// path correctness still requires GAP-1/2/3 (DRAM↔LMEM, complete_tx, tmem_cp)
// to close.

static constexpr uint32_t kMGroups    = M_GROUPS;
static constexpr uint32_t kNGroups    = N_GROUPS;
static constexpr uint32_t kKPhases    = K_PHASES;
static constexpr uint32_t kKTilesWin  = K_TILES_PER_WIN;
static constexpr uint32_t kATileCols  = A_TILE_COLS;
static constexpr uint32_t kBTileCols  = B_TILE_COLS;
static constexpr uint32_t kCTileCols  = C_TILE_COLS;

static constexpr uint32_t kACount = kMGroups * kKPhases;
static constexpr uint32_t kBBase  = kACount;
static constexpr uint32_t kCInId  = kACount + kKPhases * kNGroups;
static constexpr uint32_t kDBase  = kCInId + 1;

static inline uint32_t a_desc(uint32_t mg, uint32_t k) { return mg * kKPhases + k; }
static inline uint32_t b_desc(uint32_t k, uint32_t ng) { return kBBase + k * kNGroups + ng; }
static inline uint32_t d_desc(uint32_t mg, uint32_t ng) { return kDBase + mg * kNGroups + ng; }

static inline void load_ab(uint32_t a_tid, uint32_t b_tid) {
  (void)vt::cpabulk_tensor_ld(a_tid, /*window=*/0);
  (void)vt::cpabulk_tensor_ld(b_tid, /*window=*/1);
  vt::tcu_wait_ld();
}

static inline void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  for (uint32_t m_group = 0; m_group < kMGroups; ++m_group) {

    uint32_t handle[2];
    handle[0] = vt::tmem_alloc(arg->col_span, /*idesc_id=*/0);
    handle[1] = vt::tmem_alloc(arg->col_span, /*idesc_id=*/0);

    uint32_t idesc = vt::make_idescriptor<vt::ATYPE, vt::BTYPE, vt::fp16, vt::fp16>(0, 0);

    for (uint32_t ng = 0; ng < kNGroups; ++ng) {
      for (uint32_t batch = 0; batch < OUTPUT_BATCHES; ++batch) {
        uint32_t m_blk = batch;

        // Stage C=init for this (m_blk, ng) batch.
        (void)vt::cpabulk_tensor_ld(kCInId, /*window=*/0);
        vt::tcu_wait_ld();

        for (uint32_t n_blk = 0; n_blk < TILES_PER_BATCH; ++n_blk) {
          (void)vt::cpabulk_tensor_ld(a_desc(m_group, 0), /*window=*/0);
          (void)vt::cpabulk_tensor_ld(b_desc(0, ng),       /*window=*/1);

          for (uint32_t kp = 0; kp < kKPhases; ++kp) {
            uint32_t cur = kp & 1;
            uint32_t nxt = 1 - cur;

            if (kp + 1 < kKPhases) {
              (void)vt::cpabulk_tensor_ld(a_desc(m_group, kp + 1), /*window=*/0);
              (void)vt::cpabulk_tensor_ld(b_desc(kp + 1, ng),       /*window=*/1);
            }

            for (uint32_t ks = 0; ks < kKTilesWin; ++ks) {
              uint32_t a_tid = m_blk * kATileCols + ks;
              uint32_t b_tid = ks * kBTileCols + n_blk;
              load_ab(a_tid, b_tid);

              vt::operand_block_t op = vt::make_operand_block(handle[cur], /*b_sdesc=*/0);
              vt::tcu_mma(/*d_taddr=*/handle[cur], idesc, &op);
            }
            (void)nxt;  // double-buffer hint preserved for future restoration
          }
        }

        uint32_t last_h = (kKPhases - 1) & 1;
        vt::tcu_wait_st();
        (void)vt::cpabulk_tensor_st(d_desc(m_group, ng), /*window=*/3);
        vt::tcu_wait_st();
        (void)last_h;
      }
    }

    vt::tmem_dealloc(handle[1], arg->col_span);
    vt::tmem_dealloc(handle[0], arg->col_span);
  }
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  uint32_t grid_dim = 1;
  return vx_spawn_threads(1, &grid_dim, arg->block_dim,
                          (vx_kernel_func_cb)kernel_body, arg);
}
