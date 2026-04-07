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

static constexpr uint32_t kMTiles   = M_TILES;
static constexpr uint32_t kNTiles   = N_TILES;
static constexpr uint32_t kKPhases  = K_PHASES;
static constexpr uint32_t kWinM     = WIN_M;
static constexpr uint32_t kWinN     = WIN_N;
static constexpr uint32_t kWinK     = WIN_K;
static constexpr uint32_t kTileDim  = TILE_M;
static constexpr uint32_t kMatM     = MATRIX_M;
static constexpr uint32_t kMatN     = MATRIX_N;
static constexpr uint32_t kMatK     = MATRIX_K;

// 每个 window 数据大小 (字节)
static constexpr uint32_t kAWinBytes = kWinM * kWinK * sizeof(input_a_t);  // 256
static constexpr uint32_t kBWinBytes = kWinK * kWinN * sizeof(input_b_t);  // 256
static constexpr uint32_t kCWinBytes = kWinM * kWinN * sizeof(output_t);   // 512
static constexpr uint32_t kColSpan   = 32;  // 最小可用值: max(win_cols * elem_bytes)

static constexpr uint8_t kRoleA = 1;
static constexpr uint8_t kRoleB = 2;
static constexpr uint8_t kRoleC = 3;

// Descriptor layout (与 kernel.cpp 一致)
static constexpr uint32_t kACount = kMTiles * kKPhases;
static constexpr uint32_t kBBase  = kACount;
static constexpr uint32_t kCInId  = kACount + kKPhases * kNTiles;
static constexpr uint32_t kDBase  = kCInId + 1;
static constexpr uint32_t kTotalDescs = kDBase + kMTiles * kNTiles;

static inline uint32_t a_desc(uint32_t m, uint32_t k) { return m * kKPhases + k; }
static inline uint32_t b_desc(uint32_t k, uint32_t n) { return kBBase + k * kNTiles + n; }
static inline uint32_t d_desc(uint32_t m, uint32_t n) { return kDBase + m * kNTiles + n; }

// Device handles
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
    vx_mem_free(input_a_buf);
    vx_mem_free(input_b_buf);
    vx_mem_free(input_c_buf);
    vx_mem_free(output_buf);
    vx_mem_free(tma_desc_buf);
    vx_mem_free(mma_desc_buf);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
    device = nullptr;
  }
}

// ---------------------------------------------------------------------------
// fp22 精确参考: 计算一个 16×16 output tile
// ---------------------------------------------------------------------------
static void compute_tile_ref(float out[kTileDim][kTileDim],
                             const std::vector<uint8_t>& h_a,
                             const std::vector<uint8_t>& h_b,
                             uint32_t m_tile, uint32_t n_tile) {
  std::array<std::array<std::array<uint32_t, 8>, 8>, 4> acc = {};
  auto a_fmt = host_utils::input_fmt<input_a_t>();
  auto b_fmt = host_utils::input_fmt<input_b_t>();

  for (uint32_t k = 0; k < kKPhases; ++k) {
    const uint8_t* a_ptr = h_a.data() + (m_tile * kKPhases + k) * kAWinBytes;
    const uint8_t* b_ptr = h_b.data() + (k * kNTiles + n_tile) * kBWinBytes;

    // 16×16 A/B window 含 2 个 k8 tile，逐 tile 参考
    // 但 AMem/BMem 内部按 16×16 tile 处理 (含 2 行/2 step)
    std::vector<AMem::packet_t> a_pkt(AMem::packet_count(a_fmt));
    std::vector<BMem::packet_t> b_pkt(BMem::packet_count(b_fmt));
    for (size_t i = 0; i < a_pkt.size(); ++i)
      std::copy_n(a_ptr + i * 64, 64, a_pkt[i].begin());
    for (size_t i = 0; i < b_pkt.size(); ++i)
      std::copy_n(b_ptr + i * 64, 64, b_pkt[i].begin());

    AMem amem;
    BMem bmem;
    tensor_mem_test_utils::bulk_fill_tile_for_reference(&amem, a_fmt, a_pkt);
    tensor_mem_test_utils::bulk_fill_tile_for_reference(&bmem, b_fmt, b_pkt);

    // m16n16k8: 对于 16×16 window 内的 2 个 k8 tile,
    // AMem 每个 slot 2 lines (step_m=0,1), 逐 k8 tile 迭代
    for (uint32_t k8 = 0; k8 < kWinK / 8; ++k8) {
      // 填充 AMem/BMem slot 用于本 k8 step
      // 注意: bulk_fill 已将整个 16×16 数据填入 slot 0
      // 但我们需要按 k8 粒度读取 primitive
      for (uint32_t sm = 0; sm < 2; ++sm) {
        for (uint32_t sn = 0; sn < 2; ++sn) {
          uint16_t a_blk[8][8] = {};
          uint16_t b_blk[8][8] = {};
          uint32_t part[8][8]  = {};
          uint32_t sub = sm * 2 + sn;
          amem.read_primitive(0, sm, a_blk);
          bmem.read_primitive(0, sn, b_blk);
          host_utils::run_open_tensorcore_primitive_fp22(a_blk, b_blk, part);
          for (uint32_t i = 0; i < 8; ++i)
            for (uint32_t j = 0; j < 8; ++j)
              acc[sub][i][j] =
                  host_utils::add_fp22_raw(acc[sub][i][j], part[i][j]);
        }
      }
    }
  }

  // fp22 → fp16 via CMem dump
  CMem cmem;
  for (uint32_t sub = 0; sub < 4; ++sub) {
    uint32_t sm = sub / 2, sn = sub % 2;
    uint16_t blk[8][8] = {};
    for (uint32_t i = 0; i < 8; ++i)
      for (uint32_t j = 0; j < 8; ++j)
        blk[i][j] = fp22_to_fp16(acc[sub][i][j]);
    cmem.store_block_fp16(sm, sn, blk);
  }

  std::vector<CMem::packet_t> opkt;
  for (uint32_t st = 0; st < CMem::kRowsPerSlot; ++st) {
    std::vector<CMem::packet_t> sub_pkts;
    cmem.dump_subtile_packets(0, host_utils::output_fmt(), st, &sub_pkts);
    opkt.insert(opkt.end(), sub_pkts.begin(), sub_pkts.end());
  }
  std::vector<uint8_t> tile_bytes(kCWinBytes, 0);
  for (size_t i = 0; i < opkt.size(); ++i)
    std::copy_n(opkt[i].begin(), 64, tile_bytes.begin() + i * 64);

  std::vector<output_t> flat(kTileDim * kTileDim,
                             host_utils::encode_output(0.0f));
  host_utils::scatter_c_tile(flat, tile_bytes.data(), 0, 0);
  for (uint32_t i = 0; i < kTileDim; ++i)
    for (uint32_t j = 0; j < kTileDim; ++j)
      out[i][j] = host_utils::decode_output(flat[i * kTileDim + j]);
}

// ---------------------------------------------------------------------------
int main() {
  std::cout << "512x512x512 dense GEMM (fp8 x fp8 -> fp16)" << std::endl;
  std::cout << "window: " << kWinM << "x" << kWinN << "x" << kWinK
            << ", col_span=" << kColSpan
            << ", output-resident (ws=1)" << std::endl;

  RT_CHECK(vx_dev_open(&device));

  uint64_t isa_flags = 0;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if (!(isa_flags & VX_ISA_EXT_TCU)) {
    std::cout << "TCU not supported!" << std::endl;
    cleanup();
    return -1;
  }

  // ---- 生成输入矩阵 ----
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

  // ---- pack tiles ----
  std::cout << "packing tiles ..." << std::endl;
  std::vector<uint8_t> h_a(kMTiles * kKPhases * kAWinBytes, 0);
  std::vector<uint8_t> h_b(kKPhases * kNTiles * kBWinBytes, 0);

  for (uint32_t m = 0; m < kMTiles; ++m)
    for (uint32_t k = 0; k < kKPhases; ++k) {
      input_a_t tile[kWinM][kWinK] = {};
      for (uint32_t r = 0; r < kWinM; ++r)
        for (uint32_t c = 0; c < kWinK; ++c)
          tile[r][c] = mat_a[(m * kWinM + r) * kMatK + k * kWinK + c];
      host_utils::pack_ab_tile(h_a, (m * kKPhases + k) * kAWinBytes, tile, false);
    }

  for (uint32_t k = 0; k < kKPhases; ++k)
    for (uint32_t n = 0; n < kNTiles; ++n) {
      input_b_t tile[kWinK][kWinN] = {};
      for (uint32_t r = 0; r < kWinK; ++r)
        for (uint32_t c = 0; c < kWinN; ++c)
          tile[r][c] = mat_b[(k * kWinK + r) * kMatN + n * kWinN + c];
      host_utils::pack_ab_tile(h_b, (k * kNTiles + n) * kBWinBytes, tile, true);
    }

  // C-input: zero tile
  std::vector<uint8_t> h_c(kCWinBytes, 0);
  {
    output_t zero[kWinM][kWinN] = {};
    host_utils::pack_c_tile(h_c, 0,
        reinterpret_cast<const output_t (*)[kTileDim]>(zero));
  }

  // ---- 分配设备内存 ----
  uint32_t out_total = kMTiles * kNTiles * kCWinBytes;
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
  for (uint32_t m = 0; m < kMTiles; ++m)
    for (uint32_t k = 0; k < kKPhases; ++k) {
      auto& d = tma[a_desc(m, k)];
      d.addr = a_addr + (m * kKPhases + k) * kAWinBytes;
      d.size_bytes = kAWinBytes;
      d.tile_role = kRoleA;
    }
  for (uint32_t k = 0; k < kKPhases; ++k)
    for (uint32_t n = 0; n < kNTiles; ++n) {
      auto& d = tma[b_desc(k, n)];
      d.addr = b_addr + (k * kNTiles + n) * kBWinBytes;
      d.size_bytes = kBWinBytes;
      d.tile_role = kRoleB;
    }
  tma[kCInId].addr = c_addr;
  tma[kCInId].size_bytes = kCWinBytes;
  tma[kCInId].tile_role = kRoleC;
  for (uint32_t m = 0; m < kMTiles; ++m)
    for (uint32_t n = 0; n < kNTiles; ++n) {
      auto& d = tma[d_desc(m, n)];
      d.addr = out_addr + (m * kNTiles + n) * kCWinBytes;
      d.size_bytes = kCWinBytes;
      d.tile_role = kRoleC;  // D uses same tile_role as C
    }

  // ---- MMA descriptor: m16n16k16, ws=1 (output-resident) ----
  std::vector<mma_descriptor_t> mma(1, mma_descriptor_t{});
  mma[0].fmt_a = vt::ATYPE::id;
  mma[0].fmt_b = vt::BTYPE::id;
  mma[0].fmt_c = vt::OTYPE::id;
  mma[0].fmt_d = vt::OTYPE::id;
  mma[0].ws = 1;  // output-resident
  mma[0].a_rows = kWinM;
  mma[0].a_cols = kWinK;
  mma[0].b_rows = kWinK;
  mma[0].b_cols = kWinN;
  mma[0].c_rows = kWinM;
  mma[0].c_cols = kWinN;

  // ---- upload ----
  RT_CHECK(vx_copy_to_dev(input_a_buf, h_a.data(), 0, h_a.size()));
  RT_CHECK(vx_copy_to_dev(input_b_buf, h_b.data(), 0, h_b.size()));
  RT_CHECK(vx_copy_to_dev(input_c_buf, h_c.data(), 0, h_c.size()));

  RT_CHECK(vx_mem_alloc(device, tma.size() * sizeof(tma_descriptor_t),
                        VX_MEM_READ, &tma_desc_buf));
  RT_CHECK(vx_mem_alloc(device, mma.size() * sizeof(mma_descriptor_t),
                        VX_MEM_READ, &mma_desc_buf));
  RT_CHECK(vx_copy_to_dev(tma_desc_buf, tma.data(), 0,
                           tma.size() * sizeof(tma_descriptor_t)));
  RT_CHECK(vx_copy_to_dev(mma_desc_buf, mma.data(), 0,
                           mma.size() * sizeof(mma_descriptor_t)));

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

  // ---- verify ----
  std::cout << "verifying ..." << std::endl;
  std::vector<uint8_t> h_out(out_total, 0);
  RT_CHECK(vx_copy_from_dev(h_out.data(), output_buf, 0, out_total));

  float max_err = 0.0f;
  int errors = 0;
  constexpr float tol = 1e-2f;

  for (uint32_t m = 0; m < kMTiles; ++m)
    for (uint32_t n = 0; n < kNTiles; ++n) {
      float ref[kTileDim][kTileDim] = {};
      compute_tile_ref(ref, h_a, h_b, m, n);

      std::vector<output_t> hw(kTileDim * kTileDim,
                               host_utils::encode_output(0.0f));
      host_utils::scatter_c_tile(hw,
          h_out.data() + (m * kNTiles + n) * kCWinBytes, 0, 0);
      std::vector<float> hw_f;
      host_utils::convert_output_matrix_to_float(hw_f, hw);

      for (uint32_t i = 0; i < kTileDim; ++i)
        for (uint32_t j = 0; j < kTileDim; ++j) {
          float act = hw_f[i * kTileDim + j];
          float exp = ref[i][j];
          float e = std::fabs(act - exp);
          max_err = std::max(max_err, e);
          if (e > tol) {
            if (errors < 16)
              std::cout << "tile(" << m << "," << n << ")[" << i << ","
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
