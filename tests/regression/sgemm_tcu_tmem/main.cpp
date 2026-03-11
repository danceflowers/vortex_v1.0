#include <algorithm>
#include <chrono>
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

static constexpr uint32_t kMatrixM = 128;
static constexpr uint32_t kMatrixN = 128;
static constexpr uint32_t kMatrixK = 128;
static constexpr uint32_t kTileDim = 16;
static constexpr uint32_t kKSlice = 16;
static constexpr uint32_t kTileRows = kMatrixM / kTileDim;
static constexpr uint32_t kTileCols = kMatrixN / kTileDim;
static constexpr uint32_t kKPhases = kMatrixK / kKSlice;
static constexpr uint32_t kNumTiles = kTileRows * kTileCols;
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

static void print_matrix_block(const char* title,
                               const std::vector<float>& matrix,
                               uint32_t rows,
                               uint32_t cols,
                               bool dump_all) {
  uint32_t dump_rows = dump_all ? rows : std::min<uint32_t>(8, rows);
  uint32_t dump_cols = dump_all ? cols : std::min<uint32_t>(8, cols);
  std::cout << title << " (" << dump_rows << "x" << dump_cols << ")" << std::endl;
  for (uint32_t i = 0; i < dump_rows; ++i) {
    for (uint32_t j = 0; j < dump_cols; ++j) {
      std::cout << matrix[i * cols + j];
      if (j + 1 != dump_cols) {
        std::cout << " ";
      }
    }
    std::cout << std::endl;
  }
}

static void gather_a_tile(uint16_t tile[kTileDim][kTileDim],
                          const std::vector<uint16_t>& matrix,
                          uint32_t tile_row,
                          uint32_t k_phase) {
  uint32_t row_base = tile_row * kTileDim;
  uint32_t col_base = k_phase * kKSlice;
  for (uint32_t i = 0; i < kTileDim; ++i) {
    for (uint32_t j = 0; j < kTileDim; ++j) {
      tile[i][j] = matrix[(row_base + i) * kMatrixK + (col_base + j)];
    }
  }
}

static void gather_b_tile(uint16_t tile[kTileDim][kTileDim],
                          const std::vector<uint16_t>& matrix,
                          uint32_t k_phase,
                          uint32_t tile_col) {
  uint32_t row_base = k_phase * kKSlice;
  uint32_t col_base = tile_col * kTileDim;
  for (uint32_t i = 0; i < kTileDim; ++i) {
    for (uint32_t j = 0; j < kTileDim; ++j) {
      tile[i][j] = matrix[(row_base + i) * kMatrixN + (col_base + j)];
    }
  }
}

static void gather_c_tile(float tile[kTileDim][kTileDim],
                          const std::vector<float>& matrix,
                          uint32_t tile_row,
                          uint32_t tile_col) {
  uint32_t row_base = tile_row * kTileDim;
  uint32_t col_base = tile_col * kTileDim;
  for (uint32_t i = 0; i < kTileDim; ++i) {
    for (uint32_t j = 0; j < kTileDim; ++j) {
      tile[i][j] = matrix[(row_base + i) * kMatrixN + (col_base + j)];
    }
  }
}

static void scatter_c_tile(std::vector<float>& matrix,
                           const uint8_t* tile_bytes,
                           uint32_t tile_row,
                           uint32_t tile_col) {
  uint32_t row_base = tile_row * kTileDim;
  uint32_t col_base = tile_col * kTileDim;
  for (uint32_t i = 0; i < kTileDim; ++i) {
    for (uint32_t j = 0; j < kTileDim; ++j) {
      uint32_t off = i * 64 + j * 4;
      union {
        uint32_t u;
        float f;
      } cvt = {
        static_cast<uint32_t>(tile_bytes[off + 0])
        | (static_cast<uint32_t>(tile_bytes[off + 1]) << 8)
        | (static_cast<uint32_t>(tile_bytes[off + 2]) << 16)
        | (static_cast<uint32_t>(tile_bytes[off + 3]) << 24)
      };
      matrix[(row_base + i) * kMatrixN + (col_base + j)] = cvt.f;
    }
  }
}

static void pack_ab_tile(std::vector<uint8_t>& composite,
                         uint32_t byte_offset,
                         const uint16_t tile[kTileDim][kTileDim],
                         bool is_b) {
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

static void pack_c_tile(std::vector<uint8_t>& composite,
                        uint32_t byte_offset,
                        const float tile[kTileDim][kTileDim]) {
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

static void build_phase_input(std::vector<uint8_t>& composite,
                              const std::vector<uint16_t>& a_matrix,
                              const std::vector<uint16_t>& b_matrix,
                              const std::vector<float>& c_matrix,
                              uint32_t k_phase) {
  composite.assign(kNumTiles * kCompositeBytes, 0);
  for (uint32_t tile_row = 0; tile_row < kTileRows; ++tile_row) {
    for (uint32_t tile_col = 0; tile_col < kTileCols; ++tile_col) {
      uint32_t tile_id = tile_row * kTileCols + tile_col;
      uint32_t base = tile_id * kCompositeBytes;
      uint16_t a_tile[kTileDim][kTileDim];
      uint16_t b_tile[kTileDim][kTileDim];
      float c_tile[kTileDim][kTileDim];
      gather_a_tile(a_tile, a_matrix, tile_row, k_phase);
      gather_b_tile(b_tile, b_matrix, k_phase, tile_col);
      gather_c_tile(c_tile, c_matrix, tile_row, tile_col);
      pack_ab_tile(composite, base + 0, a_tile, false);
      pack_ab_tile(composite, base + kABBytes, b_tile, true);
      pack_c_tile(composite, base + kABBytes + kABBytes, c_tile);
    }
  }
}

static void build_quantized_ref(std::vector<float>& out,
                                const std::vector<uint16_t>& a_matrix,
                                const std::vector<uint16_t>& b_matrix) {
  out.assign(kMatrixM * kMatrixN, 0.0f);
  for (uint32_t m = 0; m < kMatrixM; ++m) {
    for (uint32_t n = 0; n < kMatrixN; ++n) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < kMatrixK; ++k) {
        float a = static_cast<float>(fp9_to_double(convert_to_fp9(a_matrix[m * kMatrixK + k], PREC_FP16)));
        float b = static_cast<float>(fp9_to_double(convert_to_fp9(b_matrix[k * kMatrixN + n], PREC_FP16)));
        sum += a * b;
      }
      out[m * kMatrixN + n] = sum;
    }
  }
}

static void build_fp32_ref(std::vector<float>& out,
                           const std::vector<uint16_t>& a_matrix,
                           const std::vector<uint16_t>& b_matrix) {
  out.assign(kMatrixM * kMatrixN, 0.0f);
  for (uint32_t m = 0; m < kMatrixM; ++m) {
    for (uint32_t n = 0; n < kMatrixN; ++n) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < kMatrixK; ++k) {
        float a = fp16_bits_to_float(a_matrix[m * kMatrixK + k]);
        float b = fp16_bits_to_float(b_matrix[k * kMatrixN + n]);
        sum += a * b;
      }
      out[m * kMatrixN + n] = sum;
    }
  }
}

int main() {
  std::srand(50);

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

  std::cout << "input data type: " << vt::ITYPE::name << " (id=" << vt::ITYPE::id << ")" << std::endl;
  std::cout << "output data type: " << vt::OTYPE::name << " (id=" << vt::OTYPE::id << ")" << std::endl;
  std::cout << "matrix A: " << kMatrixM << "x" << kMatrixK << std::endl;
  std::cout << "matrix B: " << kMatrixK << "x" << kMatrixN << std::endl;
  std::cout << "matrix D: " << kMatrixM << "x" << kMatrixN << std::endl;
  std::cout << "TMEM K slice: " << kKSlice << std::endl;

  std::vector<uint16_t> h_a(kMatrixM * kMatrixK);
  std::vector<uint16_t> h_b(kMatrixK * kMatrixN);
  for (uint32_t i = 0; i < kMatrixM; ++i) {
    for (uint32_t k = 0; k < kMatrixK; ++k) {
      int32_t raw = static_cast<int32_t>(((i * 17) + (k * 3)) % 41) - 20;
      float value = static_cast<float>(raw) * 0.0625f;
      h_a[i * kMatrixK + k] = float_to_fp16_bits(value);
    }
  }
  for (uint32_t k = 0; k < kMatrixK; ++k) {
    for (uint32_t j = 0; j < kMatrixN; ++j) {
      int32_t raw = static_cast<int32_t>(((k * 11) + (j * 5)) % 37) - 18;
      float value = static_cast<float>(raw) * 0.0625f;
      h_b[k * kMatrixN + j] = float_to_fp16_bits(value);
    }
  }

  std::vector<float> h_quant_ref;
  std::vector<float> h_fp32_ref;
  std::cout << "build OpenTensorCore reference" << std::endl;
  build_quantized_ref(h_quant_ref, h_a, h_b);
  std::cout << "build FP32 accumulate reference" << std::endl;
  build_fp32_ref(h_fp32_ref, h_a, h_b);

  std::vector<uint8_t> h_phase_input(kNumTiles * kCompositeBytes, 0);
  std::vector<uint8_t> h_phase_output(kNumTiles * kCBytes, 0);
  std::vector<float> h_accum(kMatrixM * kMatrixN, 0.0f);

  std::vector<tma_descriptor_t> in_descs(kNumTiles);
  std::vector<tma_descriptor_t> out_descs(kNumTiles);

  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, h_phase_input.size(), VX_MEM_READ, &input_buffer));
  RT_CHECK(vx_mem_alloc(device, h_phase_output.size(), VX_MEM_WRITE, &output_buffer));
  RT_CHECK(vx_mem_alloc(device, in_descs.size() * sizeof(tma_descriptor_t), VX_MEM_READ, &in_desc_buffer));
  RT_CHECK(vx_mem_alloc(device, out_descs.size() * sizeof(tma_descriptor_t), VX_MEM_READ, &out_desc_buffer));

  uint64_t input_addr = 0;
  uint64_t output_addr = 0;
  RT_CHECK(vx_mem_address(input_buffer, &input_addr));
  RT_CHECK(vx_mem_address(output_buffer, &output_addr));

  for (uint32_t tile_id = 0; tile_id < kNumTiles; ++tile_id) {
    in_descs[tile_id] = {};
    in_descs[tile_id].addr = input_addr + tile_id * kCompositeBytes;
    in_descs[tile_id].size_bytes = kCompositeBytes;
    out_descs[tile_id] = {};
    out_descs[tile_id].addr = output_addr + tile_id * kCBytes;
    out_descs[tile_id].size_bytes = kCBytes;
  }

  RT_CHECK(vx_copy_to_dev(in_desc_buffer, in_descs.data(), 0, in_descs.size() * sizeof(tma_descriptor_t)));
  RT_CHECK(vx_copy_to_dev(out_desc_buffer, out_descs.data(), 0, out_descs.size() * sizeof(tma_descriptor_t)));

  RT_CHECK(vx_mem_address(in_desc_buffer, &kernel_arg.in_desc_addr));
  RT_CHECK(vx_mem_address(out_desc_buffer, &kernel_arg.out_desc_addr));
  kernel_arg.block_dim[0] = NUM_THREADS;
  kernel_arg.block_dim[1] = 1;
  kernel_arg.tile_grid[0] = kTileCols;
  kernel_arg.tile_grid[1] = kTileRows;
  kernel_arg.bank_span = kBankSpan;

  std::cout << "upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

  auto time_start = std::chrono::high_resolution_clock::now();

  for (uint32_t phase = 0; phase < kKPhases; ++phase) {
    std::cout << "phase " << phase << "/" << (kKPhases - 1) << std::endl;
    build_phase_input(h_phase_input, h_a, h_b, h_accum, phase);
    RT_CHECK(vx_copy_to_dev(input_buffer, h_phase_input.data(), 0, h_phase_input.size()));

    std::cout << "start device" << std::endl;
    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
    std::cout << "wait for completion" << std::endl;
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

    RT_CHECK(vx_copy_from_dev(h_phase_output.data(), output_buffer, 0, h_phase_output.size()));
    for (uint32_t tile_row = 0; tile_row < kTileRows; ++tile_row) {
      for (uint32_t tile_col = 0; tile_col < kTileCols; ++tile_col) {
        uint32_t tile_id = tile_row * kTileCols + tile_col;
        scatter_c_tile(h_accum,
                       h_phase_output.data() + tile_id * kCBytes,
                       tile_row,
                       tile_col);
      }
    }
  }

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  std::cout << "verify result (print-only)" << std::endl;
  print_matrix_block("Result D", h_accum, kMatrixM, kMatrixN, false);
  print_matrix_block("OpenTensorCore ref", h_quant_ref, kMatrixM, kMatrixN, false);
  print_matrix_block("FP32 accumulate reference", h_fp32_ref, kMatrixM, kMatrixN, false);

  double max_abs_err = 0.0;
  double mse = 0.0;
  double max_abs_err_fp32 = 0.0;
  double mse_fp32 = 0.0;
  bool has_nan = false;
  for (uint32_t i = 0; i < h_accum.size(); ++i) {
    if (!std::isfinite(h_accum[i]) || !std::isfinite(h_quant_ref[i]) || !std::isfinite(h_fp32_ref[i])) {
      has_nan = true;
      continue;
    }
    double diff_q = std::fabs(h_accum[i] - h_quant_ref[i]);
    double diff_f = std::fabs(h_accum[i] - h_fp32_ref[i]);
    max_abs_err = std::max(max_abs_err, diff_q);
    max_abs_err_fp32 = std::max(max_abs_err_fp32, diff_f);
    mse += diff_q * diff_q;
    mse_fp32 += diff_f * diff_f;
  }
  double rmse = std::sqrt(mse / h_accum.size());
  double rmse_fp32 = std::sqrt(mse_fp32 / h_accum.size());
  std::cout << "vs_open_tensorcore: max_abs_err=" << max_abs_err << ", rmse=" << rmse << std::endl;
  std::cout << "vs_fp32_acc: max_abs_err=" << max_abs_err_fp32 << ", rmse=" << rmse_fp32 << std::endl;

  cleanup();
  if (has_nan) {
    std::cout << "FAILED: detected NaN in result or references" << std::endl;
    return -1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
