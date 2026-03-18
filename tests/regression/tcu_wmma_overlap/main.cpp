#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector>

#include <tensor_cfg.h>
#include <vortex.h>

#include "../tcu_host_utils.h"
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
using output_t = typename vt::OTYPE::dtype;
using host_utils = tcu_test::TileHostUtils<input_a_t, input_b_t, output_t, 16>;

static const char* kernel_file = "kernel.vxbin";

static constexpr uint32_t kTileCount = 4;
static constexpr uint32_t kTileDim = 16;
static constexpr uint32_t kABytes = kTileDim * kTileDim * sizeof(input_a_t);
static constexpr uint32_t kBBytes = kTileDim * kTileDim * sizeof(input_b_t);
static constexpr uint32_t kCBytes = kTileDim * kTileDim * sizeof(output_t);
static constexpr uint32_t kBankSizeBytes = 256;
static constexpr uint32_t kABankSpan = kABytes / kBankSizeBytes;
static constexpr uint32_t kBBankSpan = kBBytes / kBankSizeBytes;
static constexpr uint32_t kCBankSpan = kCBytes / kBankSizeBytes;
static constexpr uint8_t kTileRoleA = 1;
static constexpr uint8_t kTileRoleB = 2;
static constexpr uint8_t kTileRoleC = 3;
static constexpr uint8_t kPayloadDense = 0;
static_assert((kABytes % kBankSizeBytes) == 0, "A tile must be bank aligned");
static_assert((kBBytes % kBankSizeBytes) == 0, "B tile must be bank aligned");
static_assert((kCBytes % kBankSizeBytes) == 0, "C tile must be bank aligned");

vx_device_h device = nullptr;
vx_buffer_h input_a_buffer = nullptr;
vx_buffer_h input_b_buffer = nullptr;
vx_buffer_h input_c_buffer = nullptr;
vx_buffer_h output_buffer = nullptr;
vx_buffer_h tma_desc_buffer = nullptr;
vx_buffer_h mma_desc_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

void cleanup() {
  if (device) {
    vx_mem_free(input_a_buffer);
    vx_mem_free(input_b_buffer);
    vx_mem_free(input_c_buffer);
    vx_mem_free(output_buffer);
    vx_mem_free(tma_desc_buffer);
    vx_mem_free(mma_desc_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
    device = nullptr;
  }
}

int main() {
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
    std::cout << "Error: device warp size (" << num_threads
              << ") must match NUM_THREADS=" << NUM_THREADS << "!" << std::endl;
    cleanup();
    return -1;
  }

  std::vector<uint8_t> h_a(kTileCount * kABytes, 0);
  std::vector<uint8_t> h_b(kTileCount * kBBytes, 0);
  std::vector<uint8_t> h_c(kTileCount * kCBytes, 0);
  std::vector<float> refs(kTileCount * kTileDim * kTileDim, 0.0f);

  for (uint32_t tile_id = 0; tile_id < kTileCount; ++tile_id) {
    input_a_t a_tile[kTileDim][kTileDim];
    input_b_t b_tile[kTileDim][kTileDim];
    output_t c_tile[kTileDim][kTileDim];
    float ref_tile[kTileDim][kTileDim];

    for (uint32_t i = 0; i < kTileDim; ++i) {
      for (uint32_t j = 0; j < kTileDim; ++j) {
        float a_val = 0.0625f * float((tile_id + 1) * 3 + i + (j % 5));
        float b_val = 0.03125f * float((tile_id + 1) * 5 + j + (i % 7));
        float c_val = 0.125f * float((i == j) ? (tile_id + 1) : 0);
        a_tile[i][j] = host_utils::encode_a_input(a_val);
        b_tile[i][j] = host_utils::encode_b_input(b_val);
        c_tile[i][j] = host_utils::encode_output(c_val);
      }
    }

    host_utils::build_open_tensorcore_ref(ref_tile, a_tile, b_tile, c_tile);
    for (uint32_t i = 0; i < kTileDim; ++i) {
      for (uint32_t j = 0; j < kTileDim; ++j) {
        refs[tile_id * kTileDim * kTileDim + i * kTileDim + j] = ref_tile[i][j];
      }
    }

    host_utils::pack_ab_tile(h_a, tile_id * kABytes, a_tile, false);
    host_utils::pack_ab_tile(h_b, tile_id * kBBytes, b_tile, true);
    host_utils::pack_c_tile(h_c, tile_id * kCBytes, c_tile);
  }

  std::vector<tma_descriptor_t> tma_descs(4 * kTileCount);
  std::vector<mma_descriptor_t> mma_descs(1);

  RT_CHECK(vx_mem_alloc(device, h_a.size(), VX_MEM_READ, &input_a_buffer));
  RT_CHECK(vx_mem_alloc(device, h_b.size(), VX_MEM_READ, &input_b_buffer));
  RT_CHECK(vx_mem_alloc(device, h_c.size(), VX_MEM_READ, &input_c_buffer));
  RT_CHECK(vx_mem_alloc(device, kTileCount * kCBytes, VX_MEM_WRITE, &output_buffer));
  RT_CHECK(vx_mem_alloc(device, tma_descs.size() * sizeof(tma_descriptor_t), VX_MEM_READ, &tma_desc_buffer));
  RT_CHECK(vx_mem_alloc(device, mma_descs.size() * sizeof(mma_descriptor_t), VX_MEM_READ, &mma_desc_buffer));

  uint64_t input_a_addr = 0;
  uint64_t input_b_addr = 0;
  uint64_t input_c_addr = 0;
  uint64_t output_addr = 0;
  uint64_t tma_desc_table_addr = 0;
  uint64_t mma_desc_table_addr = 0;
  RT_CHECK(vx_mem_address(input_a_buffer, &input_a_addr));
  RT_CHECK(vx_mem_address(input_b_buffer, &input_b_addr));
  RT_CHECK(vx_mem_address(input_c_buffer, &input_c_addr));
  RT_CHECK(vx_mem_address(output_buffer, &output_addr));

  for (uint32_t tile_id = 0; tile_id < kTileCount; ++tile_id) {
    tma_descs[tile_id] = {};
    tma_descs[tile_id].addr = input_a_addr + tile_id * kABytes;
    tma_descs[tile_id].size_bytes = kABytes;
    tma_descs[tile_id].tile_role = kTileRoleA;
    tma_descs[tile_id].payload_kind = kPayloadDense;

    tma_descs[kTileCount + tile_id] = {};
    tma_descs[kTileCount + tile_id].addr = input_b_addr + tile_id * kBBytes;
    tma_descs[kTileCount + tile_id].size_bytes = kBBytes;
    tma_descs[kTileCount + tile_id].tile_role = kTileRoleB;
    tma_descs[kTileCount + tile_id].payload_kind = kPayloadDense;

    tma_descs[2 * kTileCount + tile_id] = {};
    tma_descs[2 * kTileCount + tile_id].addr = input_c_addr + tile_id * kCBytes;
    tma_descs[2 * kTileCount + tile_id].size_bytes = kCBytes;
    tma_descs[2 * kTileCount + tile_id].tile_role = kTileRoleC;
    tma_descs[2 * kTileCount + tile_id].payload_kind = kPayloadDense;

    tma_descs[3 * kTileCount + tile_id] = {};
    tma_descs[3 * kTileCount + tile_id].addr = output_addr + tile_id * kCBytes;
    tma_descs[3 * kTileCount + tile_id].size_bytes = kCBytes;
    tma_descs[3 * kTileCount + tile_id].tile_role = kTileRoleC;
    tma_descs[3 * kTileCount + tile_id].payload_kind = kPayloadDense;
  }

  mma_descs[0].fmt_a = vt::ATYPE::id;
  mma_descs[0].fmt_b = vt::BTYPE::id;
  mma_descs[0].fmt_c = vt::OTYPE::id;

  RT_CHECK(vx_copy_to_dev(input_a_buffer, h_a.data(), 0, h_a.size()));
  RT_CHECK(vx_copy_to_dev(input_b_buffer, h_b.data(), 0, h_b.size()));
  RT_CHECK(vx_copy_to_dev(input_c_buffer, h_c.data(), 0, h_c.size()));
  RT_CHECK(vx_copy_to_dev(tma_desc_buffer, tma_descs.data(), 0, tma_descs.size() * sizeof(tma_descriptor_t)));
  RT_CHECK(vx_copy_to_dev(mma_desc_buffer, mma_descs.data(), 0, mma_descs.size() * sizeof(mma_descriptor_t)));

  RT_CHECK(vx_mem_address(tma_desc_buffer, &tma_desc_table_addr));
  RT_CHECK(vx_mem_address(mma_desc_buffer, &mma_desc_table_addr));
  kernel_arg.desc_tables.magic = vt::descriptor_table_magic;
  kernel_arg.desc_tables.version = vt::descriptor_table_version;
  kernel_arg.desc_tables.tma_desc_count = static_cast<uint32_t>(tma_descs.size());
  kernel_arg.desc_tables.mma_desc_count = static_cast<uint32_t>(mma_descs.size());
  kernel_arg.desc_tables.tma_desc_addr = tma_desc_table_addr;
  kernel_arg.desc_tables.mma_desc_addr = mma_desc_table_addr;
  kernel_arg.block_dim[0] = NUM_THREADS;
  kernel_arg.block_dim[1] = 1;
  kernel_arg.a_bank_span = kABankSpan;
  kernel_arg.b_bank_span = kBBankSpan;
  kernel_arg.c_bank_span = kCBankSpan;

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  std::vector<uint8_t> h_output_bytes(kTileCount * kCBytes, 0);
  RT_CHECK(vx_copy_from_dev(h_output_bytes.data(), output_buffer, 0, h_output_bytes.size()));

  std::vector<output_t> h_output(kTileCount * kTileDim * kTileDim, host_utils::encode_output(0.0f));
  for (uint32_t tile_id = 0; tile_id < kTileCount; ++tile_id) {
    host_utils::scatter_c_tile(h_output,
                               h_output_bytes.data() + tile_id * kCBytes,
                               tile_id,
                               0);
  }

  std::vector<float> h_output_float;
  host_utils::convert_output_matrix_to_float(h_output_float, h_output);

  float max_abs_err = 0.0f;
  int errors = 0;
  constexpr float tolerance = 1e-6f;
  for (uint32_t tile_id = 0; tile_id < kTileCount; ++tile_id) {
    for (uint32_t i = 0; i < kTileDim; ++i) {
      for (uint32_t j = 0; j < kTileDim; ++j) {
        uint32_t idx = tile_id * kTileDim * kTileDim + i * kTileDim + j;
        auto actual = h_output_float[idx];
        auto expect = refs[idx];
        auto err = std::fabs(actual - expect);
        max_abs_err = std::max(max_abs_err, err);
        if (err > tolerance) {
          if (errors < 16) {
            std::cout << "tile=" << tile_id << " mismatch[" << i << "," << j << "]: actual=" << actual
                      << ", expected=" << expect << ", err=" << err << std::endl;
          }
          ++errors;
        }
      }
    }
  }

  std::cout << "max_abs_err=" << max_abs_err << std::endl;

  cleanup();

  if (errors != 0) {
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
