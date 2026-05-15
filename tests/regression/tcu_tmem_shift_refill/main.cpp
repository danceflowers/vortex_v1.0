#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <tensor_cfg.h>
#include <vortex.h>

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

static const char* kernel_file = "kernel.vxbin";

static constexpr uint32_t kRows = 16;
static constexpr uint32_t kCols = 16;
static constexpr uint32_t kRefillRows = 1;
static constexpr uint32_t kElemBytes = sizeof(uint16_t);
static constexpr uint32_t kRowBytes = kCols * kElemBytes;
static constexpr uint32_t kTileBytes = kRows * kRowBytes;
static constexpr uint32_t kRefillBytes = kRefillRows * kRowBytes;
static constexpr uint32_t kWindowColSpan = kCols * kElemBytes;
static constexpr uint8_t kTileRoleNone = 0;
static constexpr uint8_t kPayloadDense = 0;

vx_device_h device = nullptr;
vx_buffer_h input_buffer = nullptr;
vx_buffer_h refill_buffer = nullptr;
vx_buffer_h output_buffer = nullptr;
vx_buffer_h tma_desc_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

void cleanup() {
  if (device) {
    vx_mem_free(input_buffer);
    vx_mem_free(refill_buffer);
    vx_mem_free(output_buffer);
    vx_mem_free(tma_desc_buffer);
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
    std::cout << "Error: device warp size (" << num_threads << ") must match NUM_THREADS=" << NUM_THREADS << "!" << std::endl;
    cleanup();
    return -1;
  }

  std::vector<uint16_t> h_input(kRows * kCols);
  std::vector<uint16_t> h_refill(kCols);
  std::vector<uint16_t> h_expected(kRows * kCols, 0);
  for (uint32_t r = 0; r < kRows; ++r) {
    for (uint32_t c = 0; c < kCols; ++c) {
      h_input[r * kCols + c] = static_cast<uint16_t>(1 + r * kCols + c);
    }
  }
  for (uint32_t c = 0; c < kCols; ++c) {
    h_refill[c] = static_cast<uint16_t>(0x700 + c);
    h_expected[c] = h_refill[c];
  }
  for (uint32_t r = 1; r < kRows; ++r) {
    for (uint32_t c = 0; c < kCols; ++c) {
      h_expected[r * kCols + c] = h_input[(r - 1) * kCols + c];
    }
  }

  tma_descriptor_t tma_descs[3] = {};
  tma_descs[0].size_bytes = kTileBytes;
  tma_descs[0].stride_bytes = kRowBytes;
  tma_descs[0].rows = kRows;
  tma_descs[0].cols = kCols;
  tma_descs[0].elem_bytes = kElemBytes;
  tma_descs[0].tile_role = kTileRoleNone;
  tma_descs[0].payload_kind = kPayloadDense;

  tma_descs[1].size_bytes = kRefillBytes;
  tma_descs[1].stride_bytes = kRowBytes;
  tma_descs[1].rows = kRefillRows;
  tma_descs[1].cols = kCols;
  tma_descs[1].elem_bytes = kElemBytes;
  tma_descs[1].tile_role = kTileRoleNone;
  tma_descs[1].payload_kind = kPayloadDense;

  tma_descs[2].size_bytes = kTileBytes;
  tma_descs[2].stride_bytes = kRowBytes;
  tma_descs[2].rows = kRows;
  tma_descs[2].cols = kCols;
  tma_descs[2].elem_bytes = kElemBytes;
  tma_descs[2].tile_role = kTileRoleNone;
  tma_descs[2].payload_kind = kPayloadDense;

  RT_CHECK(vx_mem_alloc(device, kTileBytes, VX_MEM_READ, &input_buffer));
  RT_CHECK(vx_mem_alloc(device, kRefillBytes, VX_MEM_READ, &refill_buffer));
  RT_CHECK(vx_mem_alloc(device, kTileBytes, VX_MEM_WRITE, &output_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(tma_descs), VX_MEM_READ, &tma_desc_buffer));

  uint64_t input_addr = 0;
  uint64_t refill_addr = 0;
  uint64_t output_addr = 0;
  uint64_t tma_desc_table_addr = 0;
  RT_CHECK(vx_mem_address(input_buffer, &input_addr));
  RT_CHECK(vx_mem_address(refill_buffer, &refill_addr));
  RT_CHECK(vx_mem_address(output_buffer, &output_addr));
  tma_descs[0].addr = input_addr;
  tma_descs[1].addr = refill_addr;
  tma_descs[2].addr = output_addr;

  RT_CHECK(vx_copy_to_dev(input_buffer, h_input.data(), 0, kTileBytes));
  RT_CHECK(vx_copy_to_dev(refill_buffer, h_refill.data(), 0, kRefillBytes));
  RT_CHECK(vx_copy_to_dev(tma_desc_buffer, tma_descs, 0, sizeof(tma_descs)));

  RT_CHECK(vx_mem_address(tma_desc_buffer, &tma_desc_table_addr));
  kernel_arg.desc_tables.magic = vortex::tensor::descriptor_table_magic;
  kernel_arg.desc_tables.version = vortex::tensor::descriptor_table_version;
  kernel_arg.desc_tables.tma_desc_count = 3;
  kernel_arg.desc_tables.tma_desc_addr = tma_desc_table_addr;
  kernel_arg.grid_dim[0] = 1;
  kernel_arg.grid_dim[1] = 1;
  kernel_arg.block_dim[0] = NUM_THREADS;
  kernel_arg.block_dim[1] = 1;
  kernel_arg.bank_span = kWindowColSpan;

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  std::vector<uint16_t> h_output(kRows * kCols, 0);
  RT_CHECK(vx_copy_from_dev(h_output.data(), output_buffer, 0, kTileBytes));

  int errors = 0;
  for (uint32_t i = 0; i < h_output.size(); ++i) {
    if (h_output[i] != h_expected[i]) {
      if (errors < 16) {
        std::cout << "mismatch[" << i << "]: actual=" << h_output[i]
                  << ", expected=" << h_expected[i] << std::endl;
      }
      ++errors;
    }
  }

  cleanup();

  if (errors != 0) {
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
