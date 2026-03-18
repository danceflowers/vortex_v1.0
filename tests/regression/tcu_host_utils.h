#ifndef _TCU_HOST_UTILS_H_
#define _TCU_HOST_UTILS_H_

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <rvfloats.h>
#include <tensor_cfg.h>
#include <util.h>

#include "open_tensorcore/amem.h"
#include "open_tensorcore/bmem.h"
#include "open_tensorcore/cmem.h"
#include "open_tensorcore/fp_types.h"
#include "open_tensorcore/tensor_core_top.h"

namespace tcu_test {

template <typename InputAT, typename InputBT, typename OutputT, uint32_t TileDim>
struct TileHostUtils {
  static_assert(TileDim == 16, "only 16x16 tiles are supported");

  static constexpr uint32_t kABytes = TileDim * TileDim * sizeof(InputAT);
  static constexpr uint32_t kBBytes = TileDim * TileDim * sizeof(InputBT);
  static constexpr uint32_t kCBytes = TileDim * TileDim * sizeof(OutputT);

  static uint16_t float_to_fp16_bits(float value) {
    return rv_ftoh_s(vortex::bit_cast<uint32_t>(value), 0, nullptr);
  }

  static float fp16_bits_to_float(uint16_t value) {
    return vortex::bit_cast<float>(rv_htof_s(value, 0, nullptr));
  }

  template <typename InputT>
  static constexpr uint32_t input_fmt() {
    if constexpr (std::is_same_v<InputT, uint16_t>) {
      return vortex::tensor::fp16::id;
    } else {
      static_assert(std::is_same_v<InputT, uint8_t>, "unsupported input type");
      return vortex::tensor::fp8::id;
    }
  }

  static constexpr uint32_t output_fmt() {
    if constexpr (std::is_same_v<OutputT, float>) {
      return vortex::tensor::fp32::id;
    } else {
      static_assert(std::is_same_v<OutputT, uint16_t>, "unsupported output type");
      return vortex::tensor::fp16::id;
    }
  }

  template <typename InputT>
  static InputT encode_input(float value) {
    if constexpr (std::is_same_v<InputT, uint16_t>) {
      return float_to_fp16_bits(value);
    } else {
      return double_to_fp8_e4m3(value);
    }
  }

  template <typename InputT>
  static float decode_input(InputT value) {
    if constexpr (std::is_same_v<InputT, uint16_t>) {
      return fp16_bits_to_float(value);
    } else {
      return static_cast<float>(fp8_e4m3_to_double(value));
    }
  }

  static InputAT encode_a_input(float value) {
    return encode_input<InputAT>(value);
  }

  static InputBT encode_b_input(float value) {
    return encode_input<InputBT>(value);
  }

  static float decode_a_input(InputAT value) {
    return decode_input<InputAT>(value);
  }

  static float decode_b_input(InputBT value) {
    return decode_input<InputBT>(value);
  }

  static OutputT encode_output(float value) {
    if constexpr (std::is_same_v<OutputT, float>) {
      return value;
    } else {
      return float_to_fp16_bits(value);
    }
  }

  static float decode_output(OutputT value) {
    if constexpr (std::is_same_v<OutputT, float>) {
      return value;
    } else {
      return fp16_bits_to_float(value);
    }
  }

  static void convert_output_matrix_to_float(std::vector<float>& dst,
                                             const std::vector<OutputT>& src) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
      dst[i] = decode_output(src[i]);
    }
  }

  template <typename InputT>
  static void pack_ab_tile(std::vector<uint8_t>& composite,
                           uint32_t byte_offset,
                           const InputT tile[TileDim][TileDim],
                           bool is_b) {
    constexpr uint32_t packets_per_line = sizeof(InputT);
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

  static void pack_c_tile_fp32(std::vector<uint8_t>& composite,
                               uint32_t byte_offset,
                               const float tile[TileDim][TileDim]) {
    for (uint32_t row = 0; row < TileDim; ++row) {
      uint32_t row_offset = byte_offset + row * 64;
      for (uint32_t col = 0; col < TileDim; ++col) {
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
                               const uint16_t tile[TileDim][TileDim]) {
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

  template <typename TileT>
  static void pack_c_tile(std::vector<uint8_t>& composite,
                          uint32_t byte_offset,
                          const TileT tile[TileDim][TileDim]) {
    if constexpr (std::is_same_v<TileT, float>) {
      pack_c_tile_fp32(composite, byte_offset, tile);
    } else {
      static_assert(std::is_same_v<TileT, uint16_t>, "unsupported output tile type");
      pack_c_tile_fp16(composite, byte_offset, tile);
    }
  }

  static void scatter_c_tile_fp32(std::vector<float>& matrix,
                                  const uint8_t* tile_bytes,
                                  uint32_t tile_row,
                                  uint32_t tile_col) {
    uint32_t row_base = tile_row * TileDim;
    uint32_t col_base = tile_col * TileDim;
    for (uint32_t i = 0; i < TileDim; ++i) {
      for (uint32_t j = 0; j < TileDim; ++j) {
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
        matrix[(row_base + i) * TileDim + (col_base + j)] = cvt.f;
      }
    }
  }

  static void scatter_c_tile_fp16(std::vector<uint16_t>& matrix,
                                  const uint8_t* tile_bytes,
                                  uint32_t tile_row,
                                  uint32_t tile_col) {
    uint32_t row_base = tile_row * TileDim;
    uint32_t col_base = tile_col * TileDim;
    for (uint32_t packet = 0; packet < 8; ++packet) {
      for (uint32_t elem = 0; elem < 32; ++elem) {
        uint32_t row = packet * 2 + (elem / 16);
        uint32_t col = elem % 16;
        uint32_t off = packet * 64 + elem * 2;
        uint16_t value = static_cast<uint16_t>(tile_bytes[off + 0] | (tile_bytes[off + 1] << 8));
        matrix[(row_base + row) * TileDim + (col_base + col)] = value;
      }
    }
  }

  template <typename TileT>
  static void scatter_c_tile(std::vector<TileT>& matrix,
                             const uint8_t* tile_bytes,
                             uint32_t tile_row,
                             uint32_t tile_col) {
    if constexpr (std::is_same_v<TileT, float>) {
      scatter_c_tile_fp32(matrix, tile_bytes, tile_row, tile_col);
    } else {
      static_assert(std::is_same_v<TileT, uint16_t>, "unsupported output matrix type");
      scatter_c_tile_fp16(matrix, tile_bytes, tile_row, tile_col);
    }
  }

  static void run_open_tensorcore_primitive_fp16(const uint16_t a_in[8][8],
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

  static void build_open_tensorcore_ref(float out[TileDim][TileDim],
                                        const InputAT a_tile[TileDim][TileDim],
                                        const InputBT b_tile[TileDim][TileDim],
                                        const OutputT c_tile[TileDim][TileDim]) {
    std::vector<uint8_t> composite(kABytes + kBBytes + kCBytes, 0);
    pack_ab_tile(composite, 0, a_tile, false);
    pack_ab_tile(composite, kABytes, b_tile, true);
    pack_c_tile(composite, kABytes + kBBytes, c_tile);

    std::vector<AMem::packet_t> a_packets(AMem::packet_count(input_fmt<InputAT>()));
    std::vector<BMem::packet_t> b_packets(BMem::packet_count(input_fmt<InputBT>()));
    std::vector<CMem::packet_t> c_packets(CMem::packet_count(output_fmt()));

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
    amem.fill_tile(input_fmt<InputAT>(), a_packets);
    bmem.fill_tile(input_fmt<InputBT>(), b_packets);
    cmem.fill_tile(output_fmt(), c_packets);

    for (uint32_t storage_k = 0; storage_k < 2; ++storage_k) {
      for (uint32_t storage_m = 0; storage_m < 2; ++storage_m) {
        for (uint32_t storage_n = 0; storage_n < 2; ++storage_n) {
          uint16_t a_block[8][8] = {};
          uint16_t b_block[8][8] = {};
          amem.read_primitive(storage_m, storage_k, a_block);
          bmem.read_primitive(storage_k, storage_n, b_block);
          if constexpr (std::is_same_v<OutputT, float>) {
            float c_block[8][8] = {};
            cmem.load_block_fp32(storage_m, storage_n, c_block);
            run_open_tensorcore_primitive_fp32(a_block, b_block, c_block, c_block);
            cmem.store_block_fp32(storage_m, storage_n, c_block);
          } else {
            uint16_t c_block[8][8] = {};
            cmem.load_block_fp16(storage_m, storage_n, c_block);
            run_open_tensorcore_primitive_fp16(a_block, b_block, c_block, c_block);
            cmem.store_block_fp16(storage_m, storage_n, c_block);
          }
        }
      }
    }

    std::vector<CMem::packet_t> out_packets;
    cmem.dump_tile(output_fmt(), &out_packets);
    std::vector<uint8_t> tile_bytes(kCBytes, 0);
    for (size_t i = 0; i < out_packets.size(); ++i) {
      std::copy_n(out_packets[i].begin(), 64, tile_bytes.begin() + i * 64);
    }

    std::vector<OutputT> accum(TileDim * TileDim, encode_output(0.0f));
    scatter_c_tile(accum, tile_bytes.data(), 0, 0);
    for (uint32_t i = 0; i < TileDim; ++i) {
      for (uint32_t j = 0; j < TileDim; ++j) {
        out[i][j] = decode_output(accum[i * TileDim + j]);
      }
    }
  }
};

} // namespace tcu_test

#endif
