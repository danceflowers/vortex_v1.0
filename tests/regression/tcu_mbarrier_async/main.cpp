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

// m16n16k8 tile dimensions
static constexpr uint32_t kTileDim = 16;
static constexpr uint32_t kTileK = 8;  // K=8 for m16n16k8
static constexpr uint32_t kABytes = kTileDim * kTileK * sizeof(input_a_t);   // A is 16×8
static constexpr uint32_t kBBytes = kTileK * kTileDim * sizeof(input_b_t);   // B is 8×16
static constexpr uint32_t kCBytes = kTileDim * kTileDim * sizeof(output_t);  // C/D is 16×16
static constexpr uint32_t kCompositeBytes = kABytes + kBBytes + kCBytes;
static constexpr uint32_t align_up(uint32_t value, uint32_t align) {
  return ((value + align - 1) / align) * align;
}
static constexpr uint32_t max3(uint32_t a, uint32_t b, uint32_t c) {
  return std::max(a, std::max(b, c));
}
static constexpr uint32_t kABankSpan = align_up(kTileK * sizeof(input_a_t), 16);   // A 每行 K=8 个元素
static constexpr uint32_t kBBankSpan = align_up(kTileDim * sizeof(input_b_t), 16); // B 每行 N=16 个元素
static constexpr uint32_t kCBankSpan = align_up(kTileDim * sizeof(output_t), 16);  // C 每行 N=16 个元素
static constexpr uint32_t kBankSpan = max3(kABankSpan, kBBankSpan, kCBankSpan);
static constexpr uint8_t kTileRoleA = 1;
static constexpr uint8_t kTileRoleB = 2;
static constexpr uint8_t kTileRoleC = 3;
static constexpr uint8_t kTileRoleD = 4;
static constexpr uint8_t kPayloadDense = 0;

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

  // m16n16k8: A is 16×8, B is 8×16, C/D is 16×16
  input_a_t a_tile[kTileDim][kTileK];
  input_b_t b_tile[kTileK][kTileDim];
  output_t c_tile[kTileDim][kTileDim];
  float ref_tile[kTileDim][kTileDim];

  for (uint32_t i = 0; i < kTileDim; ++i) {
    for (uint32_t j = 0; j < kTileK; ++j) {
      a_tile[i][j] = host_utils::encode_a_input(0.125f * float((i + 1) + (j % 5)));
    }
  }
  for (uint32_t i = 0; i < kTileK; ++i) {
    for (uint32_t j = 0; j < kTileDim; ++j) {
      b_tile[i][j] = host_utils::encode_b_input(0.0625f * float((j + 1) + (i % 7)));
    }
  }
  for (uint32_t i = 0; i < kTileDim; ++i) {
    for (uint32_t j = 0; j < kTileDim; ++j) {
      c_tile[i][j] = host_utils::encode_output(0.25f * float((i == j) ? 1 : 0));
    }
  }

  // m16n16k8 reference: D[i][j] = sum_k(A[i][k]*B[k][j]) + C[i][j], k=0..7
  for (uint32_t i = 0; i < kTileDim; ++i) {
    for (uint32_t j = 0; j < kTileDim; ++j) {
      float sum = host_utils::decode_output(c_tile[i][j]);
      for (uint32_t k = 0; k < kTileK; ++k) {
        sum += host_utils::decode_a_input(a_tile[i][k]) * host_utils::decode_b_input(b_tile[k][j]);
      }
      ref_tile[i][j] = sum;
    }
  }

  std::vector<uint8_t> h_composite(kCompositeBytes, 0);
  std::memcpy(h_composite.data(), a_tile, kABytes);
  std::memcpy(h_composite.data() + kABytes, b_tile, kBBytes);
  std::memcpy(h_composite.data() + kABytes + kBBytes, c_tile, kCBytes);

  tma_descriptor_t tma_descs[4] = {};
  mma_descriptor_t mma_descs[1] = {};
  tma_descs[0].size_bytes = kABytes;
  tma_descs[0].rows = kTileDim;   // A rows = 16
  tma_descs[0].cols = kTileK;     // A cols = 8  (m16n16k8)
  tma_descs[0].elem_bytes = sizeof(input_a_t);
  tma_descs[0].tile_role = kTileRoleA;
  tma_descs[0].payload_kind = kPayloadDense;
  tma_descs[1].size_bytes = kBBytes;
  tma_descs[1].rows = kTileK;     // B rows = 8  (m16n16k8)
  tma_descs[1].cols = kTileDim;   // B cols = 16
  tma_descs[1].elem_bytes = sizeof(input_b_t);
  tma_descs[1].tile_role = kTileRoleB;
  tma_descs[1].payload_kind = kPayloadDense;
  tma_descs[2].size_bytes = kCBytes;
  tma_descs[2].rows = kTileDim;
  tma_descs[2].cols = kTileDim;
  tma_descs[2].elem_bytes = sizeof(output_t);
  tma_descs[2].tile_role = kTileRoleC;
  tma_descs[2].payload_kind = kPayloadDense;
  tma_descs[3].size_bytes = kCBytes;
  tma_descs[3].rows = kTileDim;
  tma_descs[3].cols = kTileDim;
  tma_descs[3].elem_bytes = sizeof(output_t);
  tma_descs[3].tile_role = kTileRoleD;
  tma_descs[3].payload_kind = kPayloadDense;
  mma_descs[0].fmt_a = vt::ATYPE::id;
  mma_descs[0].fmt_b = vt::BTYPE::id;
  mma_descs[0].fmt_c = vt::OTYPE::id;
  mma_descs[0].fmt_d = vt::OTYPE::id;
  mma_descs[0].a_rows = kTileDim;   // 16
  mma_descs[0].a_cols = kTileK;     // 8
  mma_descs[0].b_rows = kTileK;     // 8
  mma_descs[0].b_cols = kTileDim;   // 16
  mma_descs[0].c_rows = kTileDim;
  mma_descs[0].c_cols = kTileDim;

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
  tma_descs[1].addr = input_addr + kABytes;
  tma_descs[2].addr = input_addr + kABytes + kBBytes;
  tma_descs[3].addr = output_addr;

  RT_CHECK(vx_copy_to_dev(input_buffer, h_composite.data(), 0, h_composite.size()));
  RT_CHECK(vx_copy_to_dev(tma_desc_buffer, tma_descs, 0, sizeof(tma_descs)));
  RT_CHECK(vx_copy_to_dev(mma_desc_buffer, mma_descs, 0, sizeof(mma_descs)));

  RT_CHECK(vx_mem_address(tma_desc_buffer, &tma_desc_table_addr));
  RT_CHECK(vx_mem_address(mma_desc_buffer, &mma_desc_table_addr));
  kernel_arg.desc_tables.magic = vt::descriptor_table_magic;
  kernel_arg.desc_tables.version = vt::descriptor_table_version;
  kernel_arg.desc_tables.tma_desc_count = 4;
  kernel_arg.desc_tables.mma_desc_count = 1;
  kernel_arg.desc_tables.tma_desc_addr = tma_desc_table_addr;
  kernel_arg.desc_tables.mma_desc_addr = mma_desc_table_addr;
  kernel_arg.grid_dim[0] = 1;
  kernel_arg.grid_dim[1] = 1;
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

  std::vector<uint8_t> h_output_bytes(kCBytes, 0);
  RT_CHECK(vx_copy_from_dev(h_output_bytes.data(), output_buffer, 0, h_output_bytes.size()));

  std::vector<output_t> h_output(kTileDim * kTileDim, host_utils::encode_output(0.0f));
  std::memcpy(h_output.data(), h_output_bytes.data(), kCBytes);

  std::vector<float> h_output_float;
  host_utils::convert_output_matrix_to_float(h_output_float, h_output);

  float max_abs_err = 0.0f;
  int errors = 0;
  // fp9 精度的 tensor core 在 m16n16k8 累加后预期 ~0.5 的最大绝对误差
  constexpr float tolerance = 0.5f;
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
