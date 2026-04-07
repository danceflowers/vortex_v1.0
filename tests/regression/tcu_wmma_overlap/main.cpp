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

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------
static constexpr uint32_t kMTiles   = M_TILES;   // 32
static constexpr uint32_t kNTiles   = N_TILES;   // 32
static constexpr uint32_t kKPhases  = K_PHASES;  // 32
static constexpr uint32_t kTileDim  = TILE_M;    // 16
static constexpr uint32_t kAKDim   = TILE_K;    // 16
static constexpr uint32_t kBKDim   = TILE_K;    // 16
static constexpr uint32_t kABytes  = kTileDim * kAKDim * sizeof(input_a_t);
static constexpr uint32_t kBBytes  = kBKDim * kTileDim * sizeof(input_b_t);
static constexpr uint32_t kCBytes  = kTileDim * kTileDim * sizeof(output_t);
static constexpr uint32_t kABankSpan = kAKDim * sizeof(input_a_t);
static constexpr uint32_t kBBankSpan = kTileDim * sizeof(input_b_t);
static constexpr uint32_t kCBankSpan = kTileDim * sizeof(output_t);
static constexpr uint32_t kTmemPayloadBanks = 128;

static constexpr uint8_t kTileRoleA = 1;
static constexpr uint8_t kTileRoleB = 2;
static constexpr uint8_t kTileRoleC = 3;
static constexpr uint8_t kPayloadDense = 0;

static constexpr uint32_t kMatrixM = MATRIX_M;
static constexpr uint32_t kMatrixN = MATRIX_N;
static constexpr uint32_t kMatrixK = MATRIX_K;

// ---------------------------------------------------------------------------
// Descriptor-table layout (must match kernel.cpp)
// ---------------------------------------------------------------------------
static constexpr uint32_t kADescCount   = kMTiles * kKPhases;
static constexpr uint32_t kBDescBase    = kADescCount;
static constexpr uint32_t kBDescCount   = kKPhases * kNTiles;
static constexpr uint32_t kCInDescId    = kADescCount + kBDescCount;
static constexpr uint32_t kCOutDescBase = kCInDescId + 1;
static constexpr uint32_t kTotalTmaDescs = kCOutDescBase + kMTiles * kNTiles;

static inline uint32_t a_desc_id(uint32_t m, uint32_t k) {
  return m * kKPhases + k;
}
static inline uint32_t b_desc_id(uint32_t k, uint32_t n) {
  return kBDescBase + k * kNTiles + n;
}
static inline uint32_t c_out_desc_id(uint32_t m, uint32_t n) {
  return kCOutDescBase + m * kNTiles + n;
}

// TMEM budget check
static constexpr uint32_t required_payload_banks() {
  return 2 * kABankSpan + 2 * kBBankSpan + 2 * kCBankSpan;
}

// ---------------------------------------------------------------------------
// Device handles
// ---------------------------------------------------------------------------
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
// Reference: compute one 16x16 output tile using fp22 accumulation,
// matching the hardware's internal precision path exactly.
// ---------------------------------------------------------------------------
static void compute_tile_ref(float out[kTileDim][kTileDim],
                             const std::vector<uint8_t>& h_a,
                             const std::vector<uint8_t>& h_b,
                             uint32_t m_tile, uint32_t n_tile) {
  // fp22 accumulator for 4 subtiles (2x2 of 8x8)
  std::array<std::array<std::array<uint32_t, 8>, 8>, 4> acc = {};

  auto a_fmt = host_utils::input_fmt<input_a_t>();
  auto b_fmt = host_utils::input_fmt<input_b_t>();

  for (uint32_t k = 0; k < kKPhases; ++k) {
    const uint8_t* a_ptr = h_a.data() + (m_tile * kKPhases + k) * kABytes;
    const uint8_t* b_ptr = h_b.data() + (k * kNTiles + n_tile) * kBBytes;

    // reconstruct AMem / BMem from packed TMEM-format bytes
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

    for (uint32_t sk = 0; sk < 2; ++sk) {
      for (uint32_t sm = 0; sm < 2; ++sm) {
        for (uint32_t sn = 0; sn < 2; ++sn) {
          uint16_t a_blk[8][8] = {};
          uint16_t b_blk[8][8] = {};
          uint32_t part[8][8]  = {};
          uint32_t sub = sm * 2 + sn;
          amem.read_primitive(sk, sm, a_blk);
          bmem.read_primitive(sk, sn, b_blk);
          host_utils::run_open_tensorcore_primitive_fp22(a_blk, b_blk, part);
          for (uint32_t i = 0; i < 8; ++i)
            for (uint32_t j = 0; j < 8; ++j)
              acc[sub][i][j] =
                  host_utils::add_fp22_raw(acc[sub][i][j], part[i][j]);
        }
      }
    }
  }

  // fp22 → output format via CMem dump path
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
  std::vector<uint8_t> tile_bytes(kCBytes, 0);
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
// main
// ---------------------------------------------------------------------------
int main() {
  // TMEM budget sanity check
  if constexpr (required_payload_banks() > kTmemPayloadBanks) {
    std::cout << "Unsupported TMEM footprint: need "
              << required_payload_banks() << " payload banks, simx provides "
              << kTmemPayloadBanks << "." << std::endl;
    return -1;
  }

  std::cout << "512x512x512 dense GEMM  (fp8 x fp8 -> fp16)" << std::endl;
  std::cout << "tiles: M=" << kMTiles << " N=" << kNTiles
            << " K=" << kKPhases << std::endl;

  // ---- open device --------------------------------------------------------
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  uint64_t isa_flags = 0;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if ((isa_flags & VX_ISA_EXT_TCU) == 0) {
    std::cout << "TCU extension not supported!" << std::endl;
    cleanup();
    return -1;
  }
  uint64_t num_threads = 0;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));
  if (num_threads != NUM_THREADS) {
    std::cout << "Error: warp size (" << num_threads
              << ") != NUM_THREADS=" << NUM_THREADS << std::endl;
    cleanup();
    return -1;
  }

  // ---- generate input matrices (fp8) --------------------------------------
  std::cout << "generating input matrices ..." << std::endl;
  std::vector<input_a_t> mat_a(kMatrixM * kMatrixK);
  std::vector<input_b_t> mat_b(kMatrixK * kMatrixN);

  for (uint32_t i = 0; i < kMatrixM; ++i)
    for (uint32_t j = 0; j < kMatrixK; ++j) {
      float v = 0.0625f * float(((i * 7 + j * 13 + 1) % 15) + 1);
      mat_a[i * kMatrixK + j] = host_utils::encode_a_input(v);
    }
  for (uint32_t i = 0; i < kMatrixK; ++i)
    for (uint32_t j = 0; j < kMatrixN; ++j) {
      float v = 0.03125f * float(((i * 11 + j * 3 + 1) % 15) + 1);
      mat_b[i * kMatrixN + j] = host_utils::encode_b_input(v);
    }

  // ---- pack tiles into TMEM-format byte arrays ----------------------------
  std::cout << "packing tiles ..." << std::endl;
  const uint32_t total_a_tiles = kMTiles * kKPhases;
  const uint32_t total_b_tiles = kKPhases * kNTiles;

  std::vector<uint8_t> h_a(total_a_tiles * kABytes, 0);
  std::vector<uint8_t> h_b(total_b_tiles * kBBytes, 0);

  for (uint32_t m = 0; m < kMTiles; ++m) {
    for (uint32_t k = 0; k < kKPhases; ++k) {
      input_a_t tile[kTileDim][kAKDim] = {};
      for (uint32_t r = 0; r < kTileDim; ++r)
        for (uint32_t c = 0; c < kAKDim; ++c)
          tile[r][c] = mat_a[(m * kTileDim + r) * kMatrixK + k * kAKDim + c];
      uint32_t off = (m * kKPhases + k) * kABytes;
      host_utils::pack_ab_tile(h_a, off, tile, false);
    }
  }
  for (uint32_t k = 0; k < kKPhases; ++k) {
    for (uint32_t n = 0; n < kNTiles; ++n) {
      input_b_t tile[kBKDim][kTileDim] = {};
      for (uint32_t r = 0; r < kBKDim; ++r)
        for (uint32_t c = 0; c < kTileDim; ++c)
          tile[r][c] = mat_b[(k * kBKDim + r) * kMatrixN + n * kTileDim + c];
      uint32_t off = (k * kNTiles + n) * kBBytes;
      host_utils::pack_ab_tile(h_b, off, tile, true);
    }
  }

  // C-input: all-zero tile (shared by every output tile)
  std::vector<uint8_t> h_c(kCBytes, 0);
  {
    output_t zero_tile[kTileDim][kTileDim] = {};
    host_utils::pack_c_tile(h_c, 0,
        reinterpret_cast<const output_t (*)[kTileDim]>(zero_tile));
  }

  // ---- allocate device memory ---------------------------------------------
  const uint32_t out_total = kMTiles * kNTiles * kCBytes;
  RT_CHECK(vx_mem_alloc(device, h_a.size(),  VX_MEM_READ,  &input_a_buf));
  RT_CHECK(vx_mem_alloc(device, h_b.size(),  VX_MEM_READ,  &input_b_buf));
  RT_CHECK(vx_mem_alloc(device, h_c.size(),  VX_MEM_READ,  &input_c_buf));
  RT_CHECK(vx_mem_alloc(device, out_total,   VX_MEM_WRITE, &output_buf));

  uint64_t a_addr = 0, b_addr = 0, c_addr = 0, out_addr = 0;
  RT_CHECK(vx_mem_address(input_a_buf, &a_addr));
  RT_CHECK(vx_mem_address(input_b_buf, &b_addr));
  RT_CHECK(vx_mem_address(input_c_buf, &c_addr));
  RT_CHECK(vx_mem_address(output_buf,  &out_addr));

  // ---- build TMA descriptors -----------------------------------------------
  std::vector<tma_descriptor_t> tma_descs(kTotalTmaDescs, tma_descriptor_t{});

  // A descriptors
  for (uint32_t m = 0; m < kMTiles; ++m) {
    for (uint32_t k = 0; k < kKPhases; ++k) {
      auto& d = tma_descs[a_desc_id(m, k)];
      d.addr         = a_addr + (m * kKPhases + k) * kABytes;
      d.size_bytes   = kABytes;
      d.tile_role    = kTileRoleA;
      d.payload_kind = kPayloadDense;
    }
  }
  // B descriptors
  for (uint32_t k = 0; k < kKPhases; ++k) {
    for (uint32_t n = 0; n < kNTiles; ++n) {
      auto& d = tma_descs[b_desc_id(k, n)];
      d.addr         = b_addr + (k * kNTiles + n) * kBBytes;
      d.size_bytes   = kBBytes;
      d.tile_role    = kTileRoleB;
      d.payload_kind = kPayloadDense;
    }
  }
  // C-input descriptor (single zero buffer)
  {
    auto& d = tma_descs[kCInDescId];
    d.addr         = c_addr;
    d.size_bytes   = kCBytes;
    d.tile_role    = kTileRoleC;
    d.payload_kind = kPayloadDense;
  }
  // C-output descriptors
  for (uint32_t m = 0; m < kMTiles; ++m) {
    for (uint32_t n = 0; n < kNTiles; ++n) {
      auto& d = tma_descs[c_out_desc_id(m, n)];
      d.addr         = out_addr + (m * kNTiles + n) * kCBytes;
      d.size_bytes   = kCBytes;
      d.tile_role    = kTileRoleC;
      d.payload_kind = kPayloadDense;
    }
  }

  // MMA descriptor
  std::vector<mma_descriptor_t> mma_descs(1, mma_descriptor_t{});
  mma_descs[0].fmt_a = vt::ATYPE::id;
  mma_descs[0].fmt_b = vt::BTYPE::id;
  mma_descs[0].fmt_c = vt::OTYPE::id;
  mma_descs[0].fmt_d = vt::OTYPE::id;
  mma_descs[0].sparse_mode = vt::sparse_none;
  mma_descs[0].a_rows = kTileDim;
  mma_descs[0].a_cols = kAKDim;
  mma_descs[0].b_rows = kBKDim;
  mma_descs[0].b_cols = kTileDim;
  mma_descs[0].c_rows = kTileDim;
  mma_descs[0].c_cols = kTileDim;

  // ---- upload to device ---------------------------------------------------
  RT_CHECK(vx_copy_to_dev(input_a_buf, h_a.data(), 0, h_a.size()));
  RT_CHECK(vx_copy_to_dev(input_b_buf, h_b.data(), 0, h_b.size()));
  RT_CHECK(vx_copy_to_dev(input_c_buf, h_c.data(), 0, h_c.size()));

  RT_CHECK(vx_mem_alloc(device,
      tma_descs.size() * sizeof(tma_descriptor_t), VX_MEM_READ, &tma_desc_buf));
  RT_CHECK(vx_mem_alloc(device,
      mma_descs.size() * sizeof(mma_descriptor_t), VX_MEM_READ, &mma_desc_buf));
  RT_CHECK(vx_copy_to_dev(tma_desc_buf, tma_descs.data(), 0,
                           tma_descs.size() * sizeof(tma_descriptor_t)));
  RT_CHECK(vx_copy_to_dev(mma_desc_buf, mma_descs.data(), 0,
                           mma_descs.size() * sizeof(mma_descriptor_t)));

  uint64_t tma_tbl = 0, mma_tbl = 0;
  RT_CHECK(vx_mem_address(tma_desc_buf, &tma_tbl));
  RT_CHECK(vx_mem_address(mma_desc_buf, &mma_tbl));

  kernel_arg.desc_tables.magic          = vt::descriptor_table_magic;
  kernel_arg.desc_tables.version        = vt::descriptor_table_version;
  kernel_arg.desc_tables.tma_desc_count = static_cast<uint32_t>(tma_descs.size());
  kernel_arg.desc_tables.mma_desc_count = static_cast<uint32_t>(mma_descs.size());
  kernel_arg.desc_tables.tma_desc_addr  = tma_tbl;
  kernel_arg.desc_tables.mma_desc_addr  = mma_tbl;
  kernel_arg.block_dim[0] = NUM_THREADS;
  kernel_arg.block_dim[1] = 1;
  kernel_arg.a_bank_span  = kABankSpan;
  kernel_arg.b_bank_span  = kBBankSpan;
  kernel_arg.c_bank_span  = kCBankSpan;

  // ---- launch kernel ------------------------------------------------------
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // ---- read output --------------------------------------------------------
  std::vector<uint8_t> h_out(out_total, 0);
  RT_CHECK(vx_copy_from_dev(h_out.data(), output_buf, 0, out_total));

  // ---- verify against reference -------------------------------------------
  std::cout << "verifying ..." << std::endl;
  float max_abs_err = 0.0f;
  int errors = 0;
  constexpr float tolerance = 1e-2f;  // fp8→fp22→fp16 chain over 32 phases

  for (uint32_t m = 0; m < kMTiles; ++m) {
    for (uint32_t n = 0; n < kNTiles; ++n) {
      float ref[kTileDim][kTileDim] = {};
      compute_tile_ref(ref, h_a, h_b, m, n);

      std::vector<output_t> hw(kTileDim * kTileDim,
                               host_utils::encode_output(0.0f));
      host_utils::scatter_c_tile(
          hw, h_out.data() + (m * kNTiles + n) * kCBytes, 0, 0);

      std::vector<float> hw_f;
      host_utils::convert_output_matrix_to_float(hw_f, hw);

      for (uint32_t i = 0; i < kTileDim; ++i) {
        for (uint32_t j = 0; j < kTileDim; ++j) {
          uint32_t idx = i * kTileDim + j;
          float act = hw_f[idx];
          float exp = ref[i][j];
          float err = std::fabs(act - exp);
          max_abs_err = std::max(max_abs_err, err);
          if (err > tolerance) {
            if (errors < 16)
              std::cout << "tile(" << m << "," << n << ") [" << i << ","
                        << j << "]: actual=" << act << " expected=" << exp
                        << " err=" << err << std::endl;
            ++errors;
          }
        }
      }
    }
  }

  std::cout << "max_abs_err=" << max_abs_err << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "FAILED! (" << errors << " mismatches)" << std::endl;
    return errors;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
