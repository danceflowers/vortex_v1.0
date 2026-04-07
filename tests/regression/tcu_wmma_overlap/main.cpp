#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include <tensor_cfg.h>
#include <vortex.h>

#include "../tcu_host_utils.h"
#include "tensor/open_tensorcore/tensor_helper/tensor_mem_test_utils.h"
#include "common.h"

#define RT_CHECK(_expr)                                      \
  do {                                                       \
    int _ret = _expr;                                        \
    if (0 == _ret)                                           \
      break;                                                 \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
    cleanup();                                               \
    exit(-1);                                                \
  } while (false)

namespace vt = vortex::tensor;
using input_a_t = typename vt::ATYPE::dtype;
using input_b_t = typename vt::BTYPE::dtype;
using output_t  = typename vt::OTYPE::dtype;
using host_utils = tcu_test::TileHostUtils<input_a_t, input_b_t, output_t, 16>;

static const char* kernel_file = "kernel.vxbin";

static constexpr uint32_t kMGroups = M_GROUPS;
static constexpr uint32_t kNGroups = N_GROUPS;
static constexpr uint32_t kKPhases = K_PHASES;
static constexpr uint32_t kWinM    = WIN_M;
static constexpr uint32_t kWinN    = WIN_N;
static constexpr uint32_t kWinK    = WIN_K;
static constexpr uint32_t kTileM   = TILE_M;
static constexpr uint32_t kTileN   = TILE_N;
static constexpr uint32_t kTileK   = TILE_K;
static constexpr uint32_t kMatM    = MATRIX_M;
static constexpr uint32_t kMatN    = MATRIX_N;
static constexpr uint32_t kMatK    = MATRIX_K;
static constexpr uint32_t kColSpan = 32;  // 2 allocs × 32 cols = 64 (TMEM max)

// Window 数据大小
static constexpr uint32_t kAWinBytes = kWinM * kWinK * sizeof(input_a_t);  // 1024
static constexpr uint32_t kBWinBytes = kWinK * kWinN * sizeof(input_b_t);  // 1024
static constexpr uint32_t kCWinBytes = kWinM * kWinN * sizeof(output_t);   // 2048

static constexpr uint8_t kRoleA = 1, kRoleB = 2, kRoleC = 3;

// Descriptor layout
static constexpr uint32_t kACount     = kMGroups * kKPhases;
static constexpr uint32_t kBBase      = kACount;
static constexpr uint32_t kCInId      = kACount + kKPhases * kNGroups;
static constexpr uint32_t kDBase      = kCInId + 1;
static constexpr uint32_t kTotalDescs = kDBase + kMGroups * kNGroups;

static inline uint32_t a_desc(uint32_t mg, uint32_t k) { return mg * kKPhases + k; }
static inline uint32_t b_desc(uint32_t k, uint32_t ng) { return kBBase + k * kNGroups + ng; }
static inline uint32_t d_desc(uint32_t mg, uint32_t ng) { return kDBase + mg * kNGroups + ng; }

vx_device_h device       = nullptr;
vx_buffer_h input_a_buf  = nullptr;
vx_buffer_h input_b_buf  = nullptr;
vx_buffer_h input_c_buf  = nullptr;
vx_buffer_h output_buf   = nullptr;
vx_buffer_h tma_desc_buf = nullptr;
vx_buffer_h mma_desc_buf = nullptr;
vx_buffer_h krnl_buffer  = nullptr;
vx_buffer_h args_buffer  = nullptr;
kernel_arg_t kernel_arg  = {};

void cleanup() {
  if (device) {
    vx_mem_free(input_a_buf); vx_mem_free(input_b_buf);
    vx_mem_free(input_c_buf); vx_mem_free(output_buf);
    vx_mem_free(tma_desc_buf); vx_mem_free(mma_desc_buf);
    vx_mem_free(krnl_buffer); vx_mem_free(args_buffer);
    vx_dev_close(device); device = nullptr;
  }
}

// ---------------------------------------------------------------------------
// fp22 参考: 计算一个 16×16 output tile
// m_tile, n_tile 是全局 16×16 tile 索引 (0..31)
// ---------------------------------------------------------------------------
static void compute_tile_ref(float out[kTileM][kTileN],
                             const std::vector<uint8_t>& h_a,
                             const std::vector<uint8_t>& h_b,
                             uint32_t m_tile, uint32_t n_tile) {
  std::array<std::array<std::array<uint32_t, 8>, 8>, 4> acc = {};
  auto a_fmt = host_utils::input_fmt<input_a_t>();
  auto b_fmt = host_utils::input_fmt<input_b_t>();

  // m_tile 对应的 m_group 和 window 内 m_block
  uint32_t m_group = m_tile / (kWinM / kTileM);
  uint32_t m_blk   = m_tile % (kWinM / kTileM);
  uint32_t n_group = n_tile / (kWinN / kTileN);
  uint32_t n_blk   = n_tile % (kWinN / kTileN);

  for (uint32_t kp = 0; kp < kKPhases; ++kp) {
    const uint8_t* a_win = h_a.data() + (m_group * kKPhases + kp) * kAWinBytes;
    const uint8_t* b_win = h_b.data() + (kp * kNGroups + n_group) * kBWinBytes;

    // A window (32×32 fp8) 中提取 16×8 tile 数据 (m_blk, k_step)
    // B window (32×32 fp8) 中提取 8×16 tile 数据 (k_step, n_blk)
    for (uint32_t ks = 0; ks < kWinK / kTileK; ++ks) {
      // 提取 A[m_blk*16..(m_blk+1)*16, ks*8..(ks+1)*8] 从 32×32 window
      input_a_t a_tile_data[kTileM][kTileK] = {};
      for (uint32_t r = 0; r < kTileM; ++r)
        for (uint32_t c = 0; c < kTileK; ++c)
          a_tile_data[r][c] = a_win[(m_blk * kTileM + r) * kWinK + ks * kTileK + c];

      input_b_t b_tile_data[kTileK][kTileN] = {};
      for (uint32_t r = 0; r < kTileK; ++r)
        for (uint32_t c = 0; c < kTileN; ++c)
          b_tile_data[r][c] = b_win[(ks * kTileK + r) * kWinN + n_blk * kTileN + c];

      // 直接构造 k8 packet: 每个 packet = 一个 8×8 block (64 fp8 bytes)
      // A (16×8): 2 packets — packet[0]=rows 0-7, packet[1]=rows 8-15
      // B (8×16): 2 packets — packet[0]=cols 0-7, packet[1]=cols 8-15
      std::vector<AMem::packet_t> a_pkt(AMem::packet_count(a_fmt));  // 2 for fp8
      std::vector<BMem::packet_t> b_pkt(BMem::packet_count(b_fmt));
      for (uint32_t line = 0; line < 2; ++line) {
        // A: line=step_m, each line is 8 rows × 8 cols = 64 elements
        for (uint32_t r = 0; r < 8; ++r)
          for (uint32_t c = 0; c < 8; ++c)
            a_pkt[line][r * 8 + c] = a_tile_data[line * 8 + r][c];
        // B: line=step_n, each line is 8 rows × 8 cols = 64 elements
        for (uint32_t r = 0; r < 8; ++r)
          for (uint32_t c = 0; c < 8; ++c)
            b_pkt[line][r * 8 + c] = b_tile_data[r][line * 8 + c];
      }

      AMem amem; BMem bmem;
      tensor_mem_test_utils::bulk_fill_tile_for_reference(&amem, a_fmt, a_pkt);
      tensor_mem_test_utils::bulk_fill_tile_for_reference(&bmem, b_fmt, b_pkt);

      for (uint32_t sm = 0; sm < 2; ++sm)
        for (uint32_t sn = 0; sn < 2; ++sn) {
          uint16_t a_blk[8][8]={}, b_blk[8][8]={};
          uint32_t part[8][8]={};
          amem.read_primitive(0, sm, a_blk);
          bmem.read_primitive(0, sn, b_blk);
          host_utils::run_open_tensorcore_primitive_fp22(a_blk, b_blk, part);
          uint32_t sub = sm * 2 + sn;
          for (uint32_t i = 0; i < 8; ++i)
            for (uint32_t j = 0; j < 8; ++j)
              acc[sub][i][j] = host_utils::add_fp22_raw(acc[sub][i][j], part[i][j]);
        }
    }
  }

  // fp22 → fp16
  CMem cmem;
  for (uint32_t sub = 0; sub < 4; ++sub) {
    uint16_t blk[8][8]={};
    for (uint32_t i = 0; i < 8; ++i)
      for (uint32_t j = 0; j < 8; ++j)
        blk[i][j] = fp22_to_fp16(acc[sub][i][j]);
    cmem.store_block_fp16(sub / 2, sub % 2, blk);
  }
  std::vector<CMem::packet_t> opkt;
  for (uint32_t st = 0; st < CMem::kRowsPerSlot; ++st) {
    std::vector<CMem::packet_t> sp;
    cmem.dump_subtile_packets(0, host_utils::output_fmt(), st, &sp);
    opkt.insert(opkt.end(), sp.begin(), sp.end());
  }
  uint32_t out_tile_bytes = kTileM * kTileN * sizeof(output_t);
  std::vector<uint8_t> tb(out_tile_bytes, 0);
  for (size_t i = 0; i < opkt.size(); ++i)
    std::copy_n(opkt[i].begin(), 64, tb.begin() + i * 64);
  std::vector<output_t> flat(kTileM * kTileN, host_utils::encode_output(0.0f));
  host_utils::scatter_c_tile(flat, tb.data(), 0, 0);
  for (uint32_t i = 0; i < kTileM; ++i)
    for (uint32_t j = 0; j < kTileN; ++j)
      out[i][j] = host_utils::decode_output(flat[i * kTileN + j]);
}

// ---------------------------------------------------------------------------
int main() {
  std::cout << "512x512x512 dense GEMM (fp8 x fp8 -> fp16)" << std::endl;
  std::cout << "window: 32x32 (m32n32k32), col_span=" << kColSpan
            << ", output-resident (ws=1)" << std::endl;

  RT_CHECK(vx_dev_open(&device));
  uint64_t isa_flags = 0;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if (!(isa_flags & VX_ISA_EXT_TCU)) {
    std::cout << "TCU not supported!" << std::endl;
    cleanup(); return -1;
  }

  // ---- 生成输入 ----
  std::cout << "generating input ..." << std::endl;
  std::vector<input_a_t> mat_a(kMatM * kMatK);
  std::vector<input_b_t> mat_b(kMatK * kMatN);
  for (uint32_t i = 0; i < kMatM; ++i)
    for (uint32_t j = 0; j < kMatK; ++j)
      mat_a[i * kMatK + j] = host_utils::encode_a_input(
          0.0625f * float(((i * 7 + j * 13 + 1) % 15) + 1));
  for (uint32_t i = 0; i < kMatK; ++i)
    for (uint32_t j = 0; j < kMatN; ++j)
      mat_b[i * kMatN + j] = host_utils::encode_b_input(
          0.03125f * float(((i * 11 + j * 3 + 1) % 15) + 1));

  // ---- pack 32×32 windows ----
  std::cout << "packing tiles ..." << std::endl;
  std::vector<uint8_t> h_a(kMGroups * kKPhases * kAWinBytes, 0);
  std::vector<uint8_t> h_b(kKPhases * kNGroups * kBWinBytes, 0);

  for (uint32_t mg = 0; mg < kMGroups; ++mg)
    for (uint32_t kp = 0; kp < kKPhases; ++kp) {
      // A window: mat_a[mg*32..(mg+1)*32, kp*32..(kp+1)*32] → row-major
      uint32_t off = (mg * kKPhases + kp) * kAWinBytes;
      for (uint32_t r = 0; r < kWinM; ++r)
        for (uint32_t c = 0; c < kWinK; ++c)
          h_a[off + r * kWinK + c] = mat_a[(mg * kWinM + r) * kMatK + kp * kWinK + c];
    }
  for (uint32_t kp = 0; kp < kKPhases; ++kp)
    for (uint32_t ng = 0; ng < kNGroups; ++ng) {
      uint32_t off = (kp * kNGroups + ng) * kBWinBytes;
      for (uint32_t r = 0; r < kWinK; ++r)
        for (uint32_t c = 0; c < kWinN; ++c)
          h_b[off + r * kWinN + c] = mat_b[(kp * kWinK + r) * kMatN + ng * kWinN + c];
    }

  // C zero buffer (32×32 fp16)
  std::vector<uint8_t> h_c(kCWinBytes, 0);

  // ---- 分配设备内存 ----
  uint32_t out_total = kMGroups * kNGroups * kCWinBytes;
  RT_CHECK(vx_mem_alloc(device, h_a.size(), VX_MEM_READ, &input_a_buf));
  RT_CHECK(vx_mem_alloc(device, h_b.size(), VX_MEM_READ, &input_b_buf));
  RT_CHECK(vx_mem_alloc(device, h_c.size(), VX_MEM_READ, &input_c_buf));
  RT_CHECK(vx_mem_alloc(device, out_total,  VX_MEM_WRITE, &output_buf));

  uint64_t a_addr=0, b_addr=0, c_addr=0, out_addr=0;
  RT_CHECK(vx_mem_address(input_a_buf, &a_addr));
  RT_CHECK(vx_mem_address(input_b_buf, &b_addr));
  RT_CHECK(vx_mem_address(input_c_buf, &c_addr));
  RT_CHECK(vx_mem_address(output_buf,  &out_addr));

  // ---- TMA descriptors ----
  std::vector<tma_descriptor_t> tma(kTotalDescs, tma_descriptor_t{});
  for (uint32_t mg = 0; mg < kMGroups; ++mg)
    for (uint32_t kp = 0; kp < kKPhases; ++kp) {
      auto& d = tma[a_desc(mg, kp)];
      d.addr = a_addr + (mg * kKPhases + kp) * kAWinBytes;
      d.size_bytes = kAWinBytes;
      d.tile_role = kRoleA;
    }
  for (uint32_t kp = 0; kp < kKPhases; ++kp)
    for (uint32_t ng = 0; ng < kNGroups; ++ng) {
      auto& d = tma[b_desc(kp, ng)];
      d.addr = b_addr + (kp * kNGroups + ng) * kBWinBytes;
      d.size_bytes = kBWinBytes;
      d.tile_role = kRoleB;
    }
  tma[kCInId].addr = c_addr;
  tma[kCInId].size_bytes = kCWinBytes;
  tma[kCInId].tile_role = kRoleC;
  for (uint32_t mg = 0; mg < kMGroups; ++mg)
    for (uint32_t ng = 0; ng < kNGroups; ++ng) {
      auto& d = tma[d_desc(mg, ng)];
      d.addr = out_addr + (mg * kNGroups + ng) * kCWinBytes;
      d.size_bytes = kCWinBytes;
      d.tile_role = kRoleC;
    }

  // ---- MMA descriptor: m32n32k32, ws=1 ----
  std::vector<mma_descriptor_t> mma(1, mma_descriptor_t{});
  mma[0].fmt_a = vt::ATYPE::id;
  mma[0].fmt_b = vt::BTYPE::id;
  mma[0].fmt_c = vt::OTYPE::id;
  mma[0].fmt_d = vt::OTYPE::id;
  mma[0].ws = 1;
  mma[0].a_rows = kWinM; mma[0].a_cols = kWinK;
  mma[0].b_rows = kWinK; mma[0].b_cols = kWinN;
  mma[0].c_rows = kWinM; mma[0].c_cols = kWinN;

  // ---- upload ----
  RT_CHECK(vx_copy_to_dev(input_a_buf, h_a.data(), 0, h_a.size()));
  RT_CHECK(vx_copy_to_dev(input_b_buf, h_b.data(), 0, h_b.size()));
  RT_CHECK(vx_copy_to_dev(input_c_buf, h_c.data(), 0, h_c.size()));
  RT_CHECK(vx_mem_alloc(device, tma.size()*sizeof(tma_descriptor_t), VX_MEM_READ, &tma_desc_buf));
  RT_CHECK(vx_mem_alloc(device, mma.size()*sizeof(mma_descriptor_t), VX_MEM_READ, &mma_desc_buf));
  RT_CHECK(vx_copy_to_dev(tma_desc_buf, tma.data(), 0, tma.size()*sizeof(tma_descriptor_t)));
  RT_CHECK(vx_copy_to_dev(mma_desc_buf, mma.data(), 0, mma.size()*sizeof(mma_descriptor_t)));

  uint64_t tma_tbl=0, mma_tbl=0;
  RT_CHECK(vx_mem_address(tma_desc_buf, &tma_tbl));
  RT_CHECK(vx_mem_address(mma_desc_buf, &mma_tbl));

  kernel_arg.desc_tables.magic          = vt::descriptor_table_magic;
  kernel_arg.desc_tables.version        = vt::descriptor_table_version;
  kernel_arg.desc_tables.tma_desc_count = static_cast<uint32_t>(tma.size());
  kernel_arg.desc_tables.mma_desc_count = static_cast<uint32_t>(mma.size());
  kernel_arg.desc_tables.tma_desc_addr  = tma_tbl;
  kernel_arg.desc_tables.mma_desc_addr  = mma_tbl;
  kernel_arg.block_dim[0] = NUM_THREADS;
  kernel_arg.block_dim[1] = 1;
  kernel_arg.col_span = kColSpan;

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // ---- verify (逐 16×16 tile) ----
  std::cout << "verifying ..." << std::endl;
  std::vector<uint8_t> h_out(out_total, 0);
  RT_CHECK(vx_copy_from_dev(h_out.data(), output_buf, 0, out_total));

  float max_err = 0.0f;
  int errors = 0;
  constexpr float tol = 1e-2f;
  uint32_t m_tiles_total = kMatM / kTileM;  // 32
  uint32_t n_tiles_total = kMatN / kTileN;  // 32

  for (uint32_t mt = 0; mt < m_tiles_total; ++mt)
    for (uint32_t nt = 0; nt < n_tiles_total; ++nt) {
      float ref[kTileM][kTileN] = {};
      compute_tile_ref(ref, h_a, h_b, mt, nt);

      // 在 32×32 output block 中定位这个 16×16 tile
      uint32_t mg = mt / (kWinM / kTileM);
      uint32_t m_blk = mt % (kWinM / kTileM);
      uint32_t ng = nt / (kWinN / kTileN);
      uint32_t n_blk = nt % (kWinN / kTileN);
      const uint8_t* block_base = h_out.data() + (mg * kNGroups + ng) * kCWinBytes;

      // 从 32×32 fp16 output block 中提取 16×16 tile
      for (uint32_t i = 0; i < kTileM; ++i)
        for (uint32_t j = 0; j < kTileN; ++j) {
          uint32_t row = m_blk * kTileM + i;
          uint32_t col = n_blk * kTileN + j;
          uint32_t byte_off = (row * kWinN + col) * sizeof(output_t);
          output_t hw_val;
          std::memcpy(&hw_val, block_base + byte_off, sizeof(output_t));
          float act = host_utils::decode_output(hw_val);
          float exp = ref[i][j];
          float e = std::fabs(act - exp);
          max_err = std::max(max_err, e);
          if (e > tol) {
            if (errors < 16)
              std::cout << "tile(" << mt << "," << nt << ")[" << i << ","
                        << j << "]: act=" << act << " exp=" << exp
                        << " err=" << e << std::endl;
            ++errors;
          }
        }
    }

  std::cout << "max_abs_err=" << max_err << std::endl;
  cleanup();
  if (errors) {
    std::cout << "FAILED! (" << errors << " mismatches)" << std::endl;
    return errors;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
