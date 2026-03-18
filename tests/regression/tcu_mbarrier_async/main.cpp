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

static constexpr uint32_t kTileDim = 16;
static constexpr uint32_t kABytes = kTileDim * kTileDim * sizeof(input_a_t);
static constexpr uint32_t kBBytes = kTileDim * kTileDim * sizeof(input_b_t);
static constexpr uint32_t kCBytes = kTileDim * kTileDim * sizeof(output_t);
static constexpr uint32_t kCompositeBytes = kABytes + kBBytes + kCBytes;
static constexpr uint32_t kBankSizeBytes = 256;
static constexpr uint32_t kBankSpan = kCompositeBytes / kBankSizeBytes;
static constexpr uint8_t kTileRoleNone = 0;
static constexpr uint8_t kTileRoleC = 3;
static constexpr uint8_t kPayloadDense = 0;
static_assert((kCompositeBytes % kBankSizeBytes) == 0, "TMEM composite tile must be bank aligned");

vx_device_h device = nullptr;
vx_buffer_h input_buffer = nullptr;
vx_buffer_h output_buffer = nullptr;
vx_buffer_h tma_desc_buffer = nullptr;
vx_buffer_h mma_desc_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

void cleanup() {
  if (device) {
    vx_mem_free(input_buffer);
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

  input_a_t a_tile[kTileDim][kTileDim];
  input_b_t b_tile[kTileDim][kTileDim];
  output_t c_tile[kTileDim][kTileDim];
  float ref_tile[kTileDim][kTileDim];

  for (uint32_t i = 0; i < kTileDim; ++i) {
    for (uint32_t j = 0; j < kTileDim; ++j) {
      a_tile[i][j] = host_utils::encode_a_input(0.125f * float((i + 1) + (j % 5)));
      b_tile[i][j] = host_utils::encode_b_input(0.0625f * float((j + 1) + (i % 7)));
      c_tile[i][j] = host_utils::encode_output(0.25f * float((i == j) ? 1 : 0));
    }
  }

  host_utils::build_open_tensorcore_ref(ref_tile, a_tile, b_tile, c_tile);

  std::vector<uint8_t> h_composite(kCompositeBytes, 0);
  host_utils::pack_ab_tile(h_composite, 0, a_tile, false);
  host_utils::pack_ab_tile(h_composite, kABytes, b_tile, true);
  host_utils::pack_c_tile(h_composite, kABytes + kBBytes, c_tile);

  tma_descriptor_t tma_descs[2] = {};
  mma_descriptor_t mma_descs[1] = {};
  tma_descs[0].size_bytes = kCompositeBytes;
  tma_descs[0].tile_role = kTileRoleNone;
  tma_descs[0].payload_kind = kPayloadDense;
  tma_descs[1].size_bytes = kCBytes;
  tma_descs[1].tile_role = kTileRoleC;
  tma_descs[1].payload_kind = kPayloadDense;
  mma_descs[0].fmt_a = vt::ATYPE::id;
  mma_descs[0].fmt_b = vt::BTYPE::id;
  mma_descs[0].fmt_c = vt::OTYPE::id;

  RT_CHECK(vx_mem_alloc(device, kCompositeBytes, VX_MEM_READ, &input_buffer));
  RT_CHECK(vx_mem_alloc(device, kCBytes, VX_MEM_WRITE, &output_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(tma_descs), VX_MEM_READ, &tma_desc_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(mma_descs), VX_MEM_READ, &mma_desc_buffer));

  uint64_t input_addr = 0;
  uint64_t output_addr = 0;
  uint64_t tma_desc_table_addr = 0;
  uint64_t mma_desc_table_addr = 0;
  RT_CHECK(vx_mem_address(input_buffer, &input_addr));
  RT_CHECK(vx_mem_address(output_buffer, &output_addr));
  tma_descs[0].addr = input_addr;
  tma_descs[1].addr = output_addr;

  RT_CHECK(vx_copy_to_dev(input_buffer, h_composite.data(), 0, h_composite.size()));
  RT_CHECK(vx_copy_to_dev(tma_desc_buffer, tma_descs, 0, sizeof(tma_descs)));
  RT_CHECK(vx_copy_to_dev(mma_desc_buffer, mma_descs, 0, sizeof(mma_descs)));

  RT_CHECK(vx_mem_address(tma_desc_buffer, &tma_desc_table_addr));
  RT_CHECK(vx_mem_address(mma_desc_buffer, &mma_desc_table_addr));
  kernel_arg.desc_tables.magic = vt::descriptor_table_magic;
  kernel_arg.desc_tables.version = vt::descriptor_table_version;
  kernel_arg.desc_tables.tma_desc_count = 2;
  kernel_arg.desc_tables.mma_desc_count = 1;
  kernel_arg.desc_tables.tma_desc_addr = tma_desc_table_addr;
  kernel_arg.desc_tables.mma_desc_addr = mma_desc_table_addr;
  kernel_arg.grid_dim[0] = 1;
  kernel_arg.grid_dim[1] = 1;
  kernel_arg.block_dim[0] = NUM_THREADS;
  kernel_arg.block_dim[1] = 1;
  kernel_arg.bank_span = kBankSpan;

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  std::vector<uint8_t> h_output_bytes(kCBytes, 0);
  RT_CHECK(vx_copy_from_dev(h_output_bytes.data(), output_buffer, 0, h_output_bytes.size()));

  std::vector<output_t> h_output(kTileDim * kTileDim, host_utils::encode_output(0.0f));
  host_utils::scatter_c_tile(h_output, h_output_bytes.data(), 0, 0);

  std::vector<float> h_output_float;
  host_utils::convert_output_matrix_to_float(h_output_float, h_output);

  float max_abs_err = 0.0f;
  int errors = 0;
  constexpr float tolerance = 1e-6f;
  for (uint32_t i = 0; i < kTileDim; ++i) {
    for (uint32_t j = 0; j < kTileDim; ++j) {
      auto actual = h_output_float[i * kTileDim + j];
      auto expect = ref_tile[i][j];
      auto err = std::fabs(actual - expect);
      max_abs_err = std::max(max_abs_err, err);
      if (err > tolerance) {
        if (errors < 16) {
          std::cout << "mismatch[" << i << "," << j << "]: actual=" << actual
                    << ", expected=" << expect << ", err=" << err << std::endl;
        }
        ++errors;
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
