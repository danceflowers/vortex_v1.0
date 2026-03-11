#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include <rvfloats.h>
#include <tensor_cfg.h>
#include <util.h>
#include <vortex.h>

#include "common.h"
#include "open_tensorcore/fp_types.h"

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

static const char* kernel_file = "kernel.vxbin";

static constexpr uint32_t kTileDim = 16;
static constexpr uint32_t kABBytes = kTileDim * kTileDim * sizeof(uint16_t);
static constexpr uint32_t kCBytes = kTileDim * kTileDim * sizeof(float);
static constexpr uint32_t kCompositeBytes = kABBytes + kABBytes + kCBytes;
static constexpr uint32_t kBankSizeBytes = 256;
static constexpr uint32_t kBankSpan = kCompositeBytes / kBankSizeBytes;

vx_device_h device = nullptr;
vx_buffer_h input_buffer = nullptr;
vx_buffer_h output_buffer = nullptr;
vx_buffer_h in_desc_buffer = nullptr;
vx_buffer_h out_desc_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

void cleanup() {
  if (device) {
    vx_mem_free(input_buffer);
    vx_mem_free(output_buffer);
    vx_mem_free(in_desc_buffer);
    vx_mem_free(out_desc_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
    device = nullptr;
  }
}

static inline uint16_t float_to_fp16_bits(float value) {
  return rv_ftoh_s(vortex::bit_cast<uint32_t>(value), 0, nullptr);
}

static inline float fp16_bits_to_float(uint16_t value) {
  return vortex::bit_cast<float>(rv_htof_s(value, 0, nullptr));
}

static void pack_ab_tile(std::vector<uint8_t>& composite, uint32_t byte_offset, const uint16_t tile[kTileDim][kTileDim], bool is_b) {
  for (uint32_t line = 0; line < 4; ++line) {
    uint32_t outer = line / 2;
    uint32_t inner = line % 2;
    uint32_t row_base = is_b ? (inner * 8) : (outer * 8);
    uint32_t col_base = is_b ? (outer * 8) : (inner * 8);
    uint32_t line_offset = byte_offset + line * 128;
    for (uint32_t r = 0; r < 8; ++r) {
      for (uint32_t c = 0; c < 8; ++c) {
        uint16_t value = tile[row_base + r][col_base + c];
        uint32_t elem = r * 8 + c;
        composite[line_offset + elem * 2 + 0] = value & 0xff;
        composite[line_offset + elem * 2 + 1] = (value >> 8) & 0xff;
      }
    }
  }
}

static void pack_c_tile(std::vector<uint8_t>& composite, uint32_t byte_offset, const float tile[kTileDim][kTileDim]) {
  for (uint32_t row = 0; row < kTileDim; ++row) {
    uint32_t row_offset = byte_offset + row * 64;
    for (uint32_t col = 0; col < kTileDim; ++col) {
      union {
        float f;
        uint32_t u;
      } cvt = {tile[row][col]};
      composite[row_offset + col * 4 + 0] = cvt.u & 0xff;
      composite[row_offset + col * 4 + 1] = (cvt.u >> 8) & 0xff;
      composite[row_offset + col * 4 + 2] = (cvt.u >> 16) & 0xff;
      composite[row_offset + col * 4 + 3] = (cvt.u >> 24) & 0xff;
    }
  }
}

static void build_reference(float out[kTileDim][kTileDim],
                            const uint16_t a[kTileDim][kTileDim],
                            const uint16_t b[kTileDim][kTileDim],
                            const float c[kTileDim][kTileDim]) {
  for (uint32_t i = 0; i < kTileDim; ++i) {
    for (uint32_t j = 0; j < kTileDim; ++j) {
      float acc = c[i][j];
      for (uint32_t k = 0; k < kTileDim; ++k) {
        auto aq = static_cast<float>(fp9_to_double(convert_to_fp9(a[i][k], PREC_FP16)));
        auto bq = static_cast<float>(fp9_to_double(convert_to_fp9(b[k][j], PREC_FP16)));
        acc += aq * bq;
      }
      out[i][j] = acc;
    }
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

  uint16_t a_tile[kTileDim][kTileDim];
  uint16_t b_tile[kTileDim][kTileDim];
  float c_tile[kTileDim][kTileDim];
  float ref_tile[kTileDim][kTileDim];

  for (uint32_t i = 0; i < kTileDim; ++i) {
    for (uint32_t j = 0; j < kTileDim; ++j) {
      a_tile[i][j] = float_to_fp16_bits(0.125f * float((i + 1) + (j % 5)));
      b_tile[i][j] = float_to_fp16_bits(0.0625f * float((j + 1) + (i % 7)));
      c_tile[i][j] = 0.25f * float((i == j) ? 1 : 0);
    }
  }

  build_reference(ref_tile, a_tile, b_tile, c_tile);

  std::vector<uint8_t> h_composite(kCompositeBytes, 0);
  pack_ab_tile(h_composite, 0, a_tile, false);
  pack_ab_tile(h_composite, kABBytes, b_tile, true);
  pack_c_tile(h_composite, kABBytes + kABBytes, c_tile);

  tma_descriptor_t in_desc = {};
  tma_descriptor_t out_desc = {};
  in_desc.size_bytes = kCompositeBytes;
  out_desc.size_bytes = kCBytes;

  RT_CHECK(vx_mem_alloc(device, kCompositeBytes, VX_MEM_READ, &input_buffer));
  RT_CHECK(vx_mem_alloc(device, kCBytes, VX_MEM_WRITE, &output_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(tma_descriptor_t), VX_MEM_READ, &in_desc_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(tma_descriptor_t), VX_MEM_READ, &out_desc_buffer));

  RT_CHECK(vx_mem_address(input_buffer, &in_desc.addr));
  RT_CHECK(vx_mem_address(output_buffer, &out_desc.addr));

  RT_CHECK(vx_copy_to_dev(input_buffer, h_composite.data(), 0, h_composite.size()));
  RT_CHECK(vx_copy_to_dev(in_desc_buffer, &in_desc, 0, sizeof(in_desc)));
  RT_CHECK(vx_copy_to_dev(out_desc_buffer, &out_desc, 0, sizeof(out_desc)));

  RT_CHECK(vx_mem_address(in_desc_buffer, &kernel_arg.in_desc_addr));
  RT_CHECK(vx_mem_address(out_desc_buffer, &kernel_arg.out_desc_addr));
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

  std::vector<float> h_output(kTileDim * kTileDim, 0.0f);
  RT_CHECK(vx_copy_from_dev(h_output.data(), output_buffer, 0, kCBytes));

  float max_abs_err = 0.0f;
  int errors = 0;
  for (uint32_t i = 0; i < kTileDim; ++i) {
    for (uint32_t j = 0; j < kTileDim; ++j) {
      auto actual = h_output[i * kTileDim + j];
      auto expect = ref_tile[i][j];
      auto err = std::fabs(actual - expect);
      max_abs_err = std::max(max_abs_err, err);
      if (err > 1e-4f) {
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
