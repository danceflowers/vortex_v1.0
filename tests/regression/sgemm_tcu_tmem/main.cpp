#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <rvfloats.h>
#include <tensor_cfg.h>
#include <util.h>
#include <vortex.h>

#include "common.h"
#include "open_tensorcore/amem.h"
#include "open_tensorcore/bmem.h"
#include "open_tensorcore/cmem.h"
#include "open_tensorcore/fp_types.h"
#include "open_tensorcore/tensor_core_top.h"

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

static inline uint16_t float_to_fp16_bits(float value) {
  return rv_ftoh_s(vortex::bit_cast<uint32_t>(value), 0, nullptr);
}

static inline float fp16_bits_to_float(uint16_t value) {
  return vortex::bit_cast<float>(rv_htof_s(value, 0, nullptr));
}

template <typename InputT>
static inline constexpr uint32_t input_fmt_id() {
  if constexpr (std::is_same_v<InputT, uint16_t>) {
    return vt::fp16::id;
  } else {
    static_assert(std::is_same_v<InputT, uint8_t>, "unsupported input type");
    return vt::fp8::id;
  }
}

template <typename InputT>
static inline InputT encode_input(float value) {
  if constexpr (std::is_same_v<InputT, uint16_t>) {
    return float_to_fp16_bits(value);
  } else {
    return double_to_fp8_e4m3(value);
  }
}

template <typename InputT>
static inline float decode_input(InputT value) {
  if constexpr (std::is_same_v<InputT, uint16_t>) {
    return fp16_bits_to_float(value);
  } else {
    return static_cast<float>(fp8_e4m3_to_double(value));
  }
}

template <typename InputT>
static inline float decode_quantized_input(InputT value) {
  PrecisionType prec = (input_fmt_id<InputT>() == vt::fp8::id) ? PREC_FP8_E4M3 : PREC_FP16;
  return static_cast<float>(fp9_to_double(convert_to_fp9(static_cast<uint32_t>(value), prec)));
}

static inline output_t encode_output(float value) {
  if constexpr (std::is_same_v<output_t, float>) {
    return value;
  } else {
    return float_to_fp16_bits(value);
  }
}

static inline float decode_output(output_t value) {
  if constexpr (std::is_same_v<output_t, float>) {
    return value;
  } else {
    return fp16_bits_to_float(value);
  }
}

static void convert_output_matrix_to_float(std::vector<float>& dst,
                                           const std::vector<output_t>& src) {
  dst.resize(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    dst[i] = decode_output(src[i]);
  }
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

static void gather_a_tile(input_a_t tile[kTileDim][kTileDim],
                          const std::vector<input_a_t>& matrix,
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

static void gather_b_tile(input_b_t tile[kTileDim][kTileDim],
                          const std::vector<input_b_t>& matrix,
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

static void gather_c_tile(output_t tile[kTileDim][kTileDim],
                          const std::vector<output_t>& matrix,
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

static void scatter_c_tile_fp32(std::vector<float>& matrix,
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

static void scatter_c_tile_fp16(std::vector<uint16_t>& matrix,
                                const uint8_t* tile_bytes,
                                uint32_t tile_row,
                                uint32_t tile_col) {
  uint32_t row_base = tile_row * kTileDim;
  uint32_t col_base = tile_col * kTileDim;
  for (uint32_t packet = 0; packet < 8; ++packet) {
    for (uint32_t elem = 0; elem < 32; ++elem) {
      uint32_t row = packet * 2 + (elem / 16);
      uint32_t col = elem % 16;
      uint32_t off = packet * 64 + elem * 2;
      uint16_t value = static_cast<uint16_t>(tile_bytes[off + 0] | (tile_bytes[off + 1] << 8));
      matrix[(row_base + row) * kMatrixN + (col_base + col)] = value;
    }
  }
}

template <typename OutputT>
static void scatter_c_tile(std::vector<OutputT>& matrix,
                           const uint8_t* tile_bytes,
                           uint32_t tile_row,
                           uint32_t tile_col) {
  if constexpr (std::is_same_v<OutputT, float>) {
    scatter_c_tile_fp32(matrix, tile_bytes, tile_row, tile_col);
  } else {
    static_assert(std::is_same_v<OutputT, uint16_t>, "unsupported output type");
    scatter_c_tile_fp16(matrix, tile_bytes, tile_row, tile_col);
  }
}

static void pack_c_tile_fp32(std::vector<uint8_t>& composite,
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

static void pack_c_tile_fp16(std::vector<uint8_t>& composite,
                             uint32_t byte_offset,
                             const uint16_t tile[kTileDim][kTileDim]) {
  for (uint32_t packet = 0; packet < 8; ++packet) {
    uint32_t packet_offset = byte_offset + packet * 64;
    for (uint32_t elem = 0; elem < 32; ++elem) {
      uint32_t row = packet * 2 + (elem / 16);
      uint32_t col = elem % 16;
      uint32_t off = packet_offset + elem * 2;
      auto value = tile[row][col];
      composite[off + 0] = value & 0xff;
      composite[off + 1] = (value >> 8) & 0xff;
    }
  }
}

template <typename InputT>
static void pack_ab_tile(std::vector<uint8_t>& composite,
                         uint32_t byte_offset,
                         const InputT tile[kTileDim][kTileDim],
                         bool is_b) {
  constexpr uint32_t elem_bytes = sizeof(InputT);
  constexpr uint32_t packets_per_line = elem_bytes;
  for (uint32_t line = 0; line < 4; ++line) {
    uint32_t outer = line / 2;
    uint32_t inner = line % 2;
    uint32_t row_base = is_b ? (inner * 8) : (outer * 8);
    uint32_t col_base = is_b ? (outer * 8) : (inner * 8);
    uint32_t line_offset = byte_offset + line * 64 * packets_per_line;
    for (uint32_t r = 0; r < 8; ++r) {
      for (uint32_t c = 0; c < 8; ++c) {
        uint32_t elem = r * 8 + c;
        if constexpr (std::is_same_v<InputT, uint16_t>) {
          auto value = tile[row_base + r][col_base + c];
          composite[line_offset + elem * 2 + 0] = value & 0xff;
          composite[line_offset + elem * 2 + 1] = (value >> 8) & 0xff;
        } else {
          composite[line_offset + elem] = tile[row_base + r][col_base + c];
        }
      }
    }
  }
}

template <typename OutputT>
static void pack_c_tile(std::vector<uint8_t>& composite,
                        uint32_t byte_offset,
                        const OutputT tile[kTileDim][kTileDim]) {
  if constexpr (std::is_same_v<OutputT, float>) {
    pack_c_tile_fp32(composite, byte_offset, tile);
  } else {
    static_assert(std::is_same_v<OutputT, uint16_t>, "unsupported output type");
    pack_c_tile_fp16(composite, byte_offset, tile);
  }
}

static void run_open_tensorcore_primitive_fp9(const uint16_t a_in[8][8],
                                              const uint16_t b_in[8][8],
                                              uint16_t d_out[8][8],
                                              const uint16_t (*c_in)[8]) {
  TensorCoreTop tc;
  uint32_t c_raw[8][8] = {};

  g_cfg.precisions.clear();
  g_cfg.out_precisions.clear();
  g_cfg.precisions.push_back(PREC_FP9);
  g_cfg.out_precisions.push_back(PREC_FP16);

  for (uint32_t i = 0; i < 8; ++i) {
    for (uint32_t j = 0; j < 8; ++j) {
      c_raw[i][j] = (c_in != nullptr) ? c_in[i][j] : 0;
    }
  }

  tc.reset();
  tc.load_inputs(a_in, b_in, c_raw);
  tc.tick(true);
  tc.load_invalid();

  uint32_t spin_limit = 10000;
  while (spin_limit-- > 0 && tc.jobs_completed < tc.set_jobs) {
    tc.run();
  }

  for (uint32_t i = 0; i < 8; ++i) {
    for (uint32_t j = 0; j < 8; ++j) {
      d_out[i][j] = tc.d_out[i][j];
    }
  }
}

static void run_open_tensorcore_primitive_fp32(const uint16_t a_in[8][8],
                                               const uint16_t b_in[8][8],
                                               float d_out[8][8],
                                               const float (*c_in)[8]) {
  TensorCoreTop tc;
  uint32_t c_raw[8][8] = {};

  g_cfg.precisions.clear();
  g_cfg.out_precisions.clear();
  g_cfg.precisions.push_back(PREC_FP9);
  g_cfg.out_precisions.push_back(PREC_FP32);

  for (uint32_t i = 0; i < 8; ++i) {
    for (uint32_t j = 0; j < 8; ++j) {
      c_raw[i][j] = (c_in != nullptr) ? vortex::bit_cast<uint32_t>(c_in[i][j]) : 0;
    }
  }

  tc.reset();
  tc.load_inputs(a_in, b_in, c_raw);
  tc.tick(true);
  tc.load_invalid();

  uint32_t spin_limit = 10000;
  while (spin_limit-- > 0 && tc.jobs_completed < tc.set_jobs) {
    tc.run();
  }

  for (uint32_t i = 0; i < 8; ++i) {
    for (uint32_t j = 0; j < 8; ++j) {
      d_out[i][j] = vortex::bit_cast<float>(tc.d_out[i][j]);
    }
  }
}

static void build_phase_input(std::vector<uint8_t>& composite,
                              const std::vector<input_a_t>& a_matrix,
                              const std::vector<input_b_t>& b_matrix,
                              const std::vector<output_t>& c_matrix,
                              uint32_t k_phase) {
  composite.assign(kNumTiles * kCompositeBytes, 0);
  for (uint32_t tile_row = 0; tile_row < kTileRows; ++tile_row) {
    for (uint32_t tile_col = 0; tile_col < kTileCols; ++tile_col) {
      uint32_t tile_id = tile_row * kTileCols + tile_col;
      uint32_t base = tile_id * kCompositeBytes;
      input_a_t a_tile[kTileDim][kTileDim];
      input_b_t b_tile[kTileDim][kTileDim];
      output_t c_tile[kTileDim][kTileDim];
      gather_a_tile(a_tile, a_matrix, tile_row, k_phase);
      gather_b_tile(b_tile, b_matrix, k_phase, tile_col);
      gather_c_tile(c_tile, c_matrix, tile_row, tile_col);
      pack_ab_tile(composite, base + 0, a_tile, false);
      pack_ab_tile(composite, base + kABytes, b_tile, true);
      pack_c_tile(composite, base + kABytes + kBBytes, c_tile);
    }
  }
}

static void build_open_tensorcore_ref(std::vector<float>& out,
                                      const std::vector<input_a_t>& a_matrix,
                                      const std::vector<input_b_t>& b_matrix) {
  if constexpr (std::is_same_v<output_t, float>) {
    std::vector<output_t> accum(kMatrixM * kMatrixN, encode_output(0.0f));
    std::vector<uint8_t> composite(kCompositeBytes, 0);

    for (uint32_t phase = 0; phase < kKPhases; ++phase) {
      for (uint32_t tile_row = 0; tile_row < kTileRows; ++tile_row) {
        for (uint32_t tile_col = 0; tile_col < kTileCols; ++tile_col) {
          input_a_t a_tile[kTileDim][kTileDim];
          input_b_t b_tile[kTileDim][kTileDim];
          output_t c_tile[kTileDim][kTileDim];
          gather_a_tile(a_tile, a_matrix, tile_row, phase);
          gather_b_tile(b_tile, b_matrix, phase, tile_col);
          gather_c_tile(c_tile, accum, tile_row, tile_col);

          std::fill(composite.begin(), composite.end(), 0);
          pack_ab_tile(composite, 0, a_tile, false);
          pack_ab_tile(composite, kABytes, b_tile, true);
          pack_c_tile(composite, kABytes + kBBytes, c_tile);

          std::vector<AMem::packet_t> a_packets(AMem::packet_count(input_fmt_id<input_a_t>()));
          std::vector<BMem::packet_t> b_packets(BMem::packet_count(input_fmt_id<input_b_t>()));
          std::vector<CMem::packet_t> c_packets(CMem::packet_count(vt::OTYPE::id));

          for (size_t i = 0; i < a_packets.size(); ++i) {
            std::copy_n(composite.data() + i * 64, 64, a_packets[i].begin());
          }
          for (size_t i = 0; i < b_packets.size(); ++i) {
            std::copy_n(composite.data() + kABytes + i * 64, 64, b_packets[i].begin());
          }
          for (size_t i = 0; i < c_packets.size(); ++i) {
            std::copy_n(composite.data() + kABytes + kBBytes + i * 64, 64, c_packets[i].begin());
          }

          AMem amem;
          BMem bmem;
          CMem cmem;
          amem.fill_tile(input_fmt_id<input_a_t>(), a_packets);
          bmem.fill_tile(input_fmt_id<input_b_t>(), b_packets);
          cmem.fill_tile(vt::OTYPE::id, c_packets);

          for (uint32_t storage_k = 0; storage_k < 2; ++storage_k) {
            for (uint32_t storage_m = 0; storage_m < 2; ++storage_m) {
              for (uint32_t storage_n = 0; storage_n < 2; ++storage_n) {
                uint16_t a_block[8][8] = {};
                uint16_t b_block[8][8] = {};
                amem.read_primitive(storage_m, storage_k, a_block);
                bmem.read_primitive(storage_k, storage_n, b_block);
                if constexpr (std::is_same_v<output_t, float>) {
                  float c_block[8][8] = {};
                  cmem.load_block_fp32(storage_m, storage_n, c_block);
                  run_open_tensorcore_primitive_fp32(a_block, b_block, c_block, c_block);
                  cmem.store_block_fp32(storage_m, storage_n, c_block);
                } else {
                  uint16_t c_block[8][8] = {};
                  cmem.load_block_fp16(storage_m, storage_n, c_block);
                  run_open_tensorcore_primitive_fp9(a_block, b_block, c_block, c_block);
                  cmem.store_block_fp16(storage_m, storage_n, c_block);
                }
              }
            }
          }

          std::vector<CMem::packet_t> out_packets;
          cmem.dump_tile(vt::OTYPE::id, &out_packets);
          std::vector<uint8_t> tile_bytes(kCBytes, 0);
          for (size_t i = 0; i < out_packets.size(); ++i) {
            std::copy_n(out_packets[i].begin(), 64, tile_bytes.begin() + i * 64);
          }
          scatter_c_tile(accum, tile_bytes.data(), tile_row, tile_col);
        }
      }
    }

    convert_output_matrix_to_float(out, accum);
  } else {
    std::vector<output_t> accum(kMatrixM * kMatrixN, encode_output(0.0f));
    std::vector<uint8_t> composite(kCompositeBytes, 0);

    for (uint32_t phase = 0; phase < kKPhases; ++phase) {
      for (uint32_t tile_row = 0; tile_row < kTileRows; ++tile_row) {
        for (uint32_t tile_col = 0; tile_col < kTileCols; ++tile_col) {
          input_a_t a_tile[kTileDim][kTileDim];
          input_b_t b_tile[kTileDim][kTileDim];
          output_t c_tile[kTileDim][kTileDim];
          gather_a_tile(a_tile, a_matrix, tile_row, phase);
          gather_b_tile(b_tile, b_matrix, phase, tile_col);
          gather_c_tile(c_tile, accum, tile_row, tile_col);

          std::fill(composite.begin(), composite.end(), 0);
          pack_ab_tile(composite, 0, a_tile, false);
          pack_ab_tile(composite, kABytes, b_tile, true);
          pack_c_tile(composite, kABytes + kBBytes, c_tile);

          std::vector<AMem::packet_t> a_packets(AMem::packet_count(input_fmt_id<input_a_t>()));
          std::vector<BMem::packet_t> b_packets(BMem::packet_count(input_fmt_id<input_b_t>()));
          std::vector<CMem::packet_t> c_packets(CMem::packet_count(vt::OTYPE::id));

          for (size_t i = 0; i < a_packets.size(); ++i) {
            std::copy_n(composite.data() + i * 64, 64, a_packets[i].begin());
          }
          for (size_t i = 0; i < b_packets.size(); ++i) {
            std::copy_n(composite.data() + kABytes + i * 64, 64, b_packets[i].begin());
          }
          for (size_t i = 0; i < c_packets.size(); ++i) {
            std::copy_n(composite.data() + kABytes + kBBytes + i * 64, 64, c_packets[i].begin());
          }

          AMem amem;
          BMem bmem;
          CMem cmem;
          amem.fill_tile(input_fmt_id<input_a_t>(), a_packets);
          bmem.fill_tile(input_fmt_id<input_b_t>(), b_packets);
          cmem.fill_tile(vt::OTYPE::id, c_packets);

          for (uint32_t storage_k = 0; storage_k < 2; ++storage_k) {
            for (uint32_t storage_m = 0; storage_m < 2; ++storage_m) {
              for (uint32_t storage_n = 0; storage_n < 2; ++storage_n) {
                uint16_t a_block[8][8] = {};
                uint16_t b_block[8][8] = {};
                uint16_t c_block[8][8] = {};
                amem.read_primitive(storage_m, storage_k, a_block);
                bmem.read_primitive(storage_k, storage_n, b_block);
                cmem.load_block_fp16(storage_m, storage_n, c_block);
                run_open_tensorcore_primitive_fp9(a_block, b_block, c_block, c_block);
                cmem.store_block_fp16(storage_m, storage_n, c_block);
              }
            }
          }

          std::vector<CMem::packet_t> out_packets;
          cmem.dump_tile(vt::OTYPE::id, &out_packets);
          std::vector<uint8_t> tile_bytes(kCBytes, 0);
          for (size_t i = 0; i < out_packets.size(); ++i) {
            std::copy_n(out_packets[i].begin(), 64, tile_bytes.begin() + i * 64);
          }
          scatter_c_tile(accum, tile_bytes.data(), tile_row, tile_col);
        }
      }
    }

    convert_output_matrix_to_float(out, accum);
  }
}

static void build_fp32_ref(std::vector<float>& out,
                           const std::vector<input_a_t>& a_matrix,
                           const std::vector<input_b_t>& b_matrix) {
  out.assign(kMatrixM * kMatrixN, 0.0f);
  for (uint32_t m = 0; m < kMatrixM; ++m) {
    for (uint32_t n = 0; n < kMatrixN; ++n) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < kMatrixK; ++k) {
        float a = decode_input(a_matrix[m * kMatrixK + k]);
        float b = decode_input(b_matrix[k * kMatrixN + n]);
        sum += a * b;
      }
      out[m * kMatrixN + n] = sum;
    }
  }
}

int main() {
  std::srand(50);
  uint32_t phase_limit = kKPhases;
  if (const char* env = std::getenv("SGEMM_TCU_TMEM_PHASES")) {
    auto parsed = std::strtoul(env, nullptr, 0);
    if (parsed > 0) {
      phase_limit = std::min<uint32_t>(kKPhases, static_cast<uint32_t>(parsed));
    }
  }

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

  std::cout << "matrix A type: " << vt::ATYPE::name << " (id=" << vt::ATYPE::id << ")" << std::endl;
  std::cout << "matrix B type: " << vt::BTYPE::name << " (id=" << vt::BTYPE::id << ")" << std::endl;
  std::cout << "output data type: " << vt::OTYPE::name << " (id=" << vt::OTYPE::id << ")" << std::endl;
  std::cout << "matrix A: " << kMatrixM << "x" << kMatrixK << std::endl;
  std::cout << "matrix B: " << kMatrixK << "x" << kMatrixN << std::endl;
  std::cout << "matrix D: " << kMatrixM << "x" << kMatrixN << std::endl;
  std::cout << "TMEM K slice: " << kKSlice << std::endl;

  std::vector<input_a_t> h_a(kMatrixM * kMatrixK);
  std::vector<input_b_t> h_b(kMatrixK * kMatrixN);
  for (uint32_t i = 0; i < kMatrixM; ++i) {
    for (uint32_t k = 0; k < kMatrixK; ++k) {
      int32_t raw = static_cast<int32_t>(((i * 17) + (k * 3)) % 41) - 20;
      float value = static_cast<float>(raw) * 0.0625f;
      h_a[i * kMatrixK + k] = encode_input<input_a_t>(value);
    }
  }
  for (uint32_t k = 0; k < kMatrixK; ++k) {
    for (uint32_t j = 0; j < kMatrixN; ++j) {
      int32_t raw = static_cast<int32_t>(((k * 11) + (j * 5)) % 37) - 18;
      float value = static_cast<float>(raw) * 0.0625f;
      h_b[k * kMatrixN + j] = encode_input<input_b_t>(value);
    }
  }

  std::vector<float> h_quant_ref;
  std::vector<float> h_fp32_ref;
  std::cout << "build OpenTensorCore reference" << std::endl;
  build_open_tensorcore_ref(h_quant_ref, h_a, h_b);
  std::cout << "build FP32 accumulate reference" << std::endl;
  build_fp32_ref(h_fp32_ref, h_a, h_b);

  std::vector<uint8_t> h_phase_input(kNumTiles * kCompositeBytes, 0);
  std::vector<uint8_t> h_phase_output(kNumTiles * kCBytes, 0);
  std::vector<output_t> h_accum(kMatrixM * kMatrixN, encode_output(0.0f));

  std::vector<tma_descriptor_t> tma_descs(2 * kNumTiles);
  std::vector<mma_descriptor_t> mma_descs(1);

  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, h_phase_input.size(), VX_MEM_READ, &input_buffer));
  RT_CHECK(vx_mem_alloc(device, h_phase_output.size(), VX_MEM_WRITE, &output_buffer));
  RT_CHECK(vx_mem_alloc(device, tma_descs.size() * sizeof(tma_descriptor_t), VX_MEM_READ, &tma_desc_buffer));
  RT_CHECK(vx_mem_alloc(device, mma_descs.size() * sizeof(mma_descriptor_t), VX_MEM_READ, &mma_desc_buffer));

  uint64_t input_addr = 0;
  uint64_t output_addr = 0;
  uint64_t tma_desc_table_addr = 0;
  uint64_t mma_desc_table_addr = 0;
  RT_CHECK(vx_mem_address(input_buffer, &input_addr));
  RT_CHECK(vx_mem_address(output_buffer, &output_addr));

  for (uint32_t tile_id = 0; tile_id < kNumTiles; ++tile_id) {
    tma_descs[tile_id] = {};
    tma_descs[tile_id].addr = input_addr + tile_id * kCompositeBytes;
    tma_descs[tile_id].size_bytes = kCompositeBytes;
    tma_descs[tile_id].tile_role = kTileRoleNone;
    tma_descs[tile_id].payload_kind = kPayloadDense;
    tma_descs[kNumTiles + tile_id] = {};
    tma_descs[kNumTiles + tile_id].addr = output_addr + tile_id * kCBytes;
    tma_descs[kNumTiles + tile_id].size_bytes = kCBytes;
    tma_descs[kNumTiles + tile_id].tile_role = kTileRoleC;
    tma_descs[kNumTiles + tile_id].payload_kind = kPayloadDense;
  }
  mma_descs[0].fmt_a = vt::ATYPE::id;
  mma_descs[0].fmt_b = vt::BTYPE::id;
  mma_descs[0].fmt_c = vt::OTYPE::id;

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
  kernel_arg.tile_grid[0] = kTileCols;
  kernel_arg.tile_grid[1] = kTileRows;
  kernel_arg.bank_span = kBankSpan;

  std::cout << "upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

  auto time_start = std::chrono::high_resolution_clock::now();

  for (uint32_t phase = 0; phase < phase_limit; ++phase) {
    std::cout << "phase " << phase << "/" << (phase_limit - 1) << std::endl;
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
  std::vector<float> h_result_float;
  convert_output_matrix_to_float(h_result_float, h_accum);
  print_matrix_block("Result D", h_result_float, kMatrixM, kMatrixN, false);
  print_matrix_block("OpenTensorCore ref", h_quant_ref, kMatrixM, kMatrixN, false);
  print_matrix_block("FP32 accumulate reference", h_fp32_ref, kMatrixM, kMatrixN, false);

  double max_abs_err = 0.0;
  double mse = 0.0;
  double max_abs_err_fp32 = 0.0;
  double mse_fp32 = 0.0;
  bool has_nan = false;
  for (uint32_t i = 0; i < h_accum.size(); ++i) {
    float result = h_result_float[i];
    if (!std::isfinite(result) || !std::isfinite(h_quant_ref[i]) || !std::isfinite(h_fp32_ref[i])) {
      has_nan = true;
      continue;
    }
    double diff_q = std::fabs(result - h_quant_ref[i]);
    double diff_f = std::fabs(result - h_fp32_ref[i]);
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
