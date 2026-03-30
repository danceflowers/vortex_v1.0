#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector>

#include <tensor_cfg.h>
#include <vortex.h>

#include "../tcu_host_utils.h"
#include "open_tensorcore/meta_mem.h"
#include "open_tensorcore/sparse_select.h"
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

static constexpr bool kSparse2To4 = (SPARSE_MODE == 1);
static constexpr bool kSparse1To4 = (SPARSE_MODE == 2);
static constexpr bool kHasSparse = kSparse2To4 || kSparse1To4;
static constexpr uint32_t kWorkerWarps = WORKER_WARPS;
static constexpr uint32_t kTileCount = TILE_COUNT;
static constexpr uint32_t kPhaseCount = 4;
static constexpr uint32_t kTileDim = 16;
static constexpr uint32_t kAKDim = kSparse1To4 ? 64 : (kSparse2To4 ? 32 : 16);
static constexpr uint32_t kBKDim = kSparse1To4 ? 64 : (kSparse2To4 ? 32 : 16);
static constexpr uint32_t kACompressedK = 16;
static constexpr uint32_t kABytes = kTileDim * kACompressedK * sizeof(input_a_t);
static constexpr uint32_t kMetaBytes = kHasSparse ? 64 : 0;
static constexpr uint32_t kBBytes = kBKDim * kTileDim * sizeof(input_b_t);
static constexpr uint32_t kCBytes = kTileDim * kTileDim * sizeof(output_t);
static constexpr uint32_t ceil_div(uint32_t a, uint32_t b) {
  return (a + b - 1) / b;
}
static constexpr uint32_t kABankSpan = kAKDim * sizeof(input_a_t);
static constexpr uint32_t kMetaBankSpan = kHasSparse ? 1 : 0;
static constexpr uint32_t kMetaBankBase = 16;
static constexpr uint32_t kBBankSpan = kTileDim * sizeof(input_b_t);
static constexpr uint32_t kCBankSpan = kTileDim * sizeof(output_t);
static constexpr uint32_t kTmemPayloadBanks = 128;
static constexpr uint8_t kTileRoleA = 1;
static constexpr uint8_t kTileRoleB = 2;
static constexpr uint8_t kTileRoleC = 3;
static constexpr uint8_t kPayloadDense = 0;
static constexpr uint8_t kPayloadSparsePayload = 1;

static_assert(kWorkerWarps >= 1 && kWorkerWarps <= 2, "tcu_wmma_overlap supports 1 or 2 worker warps");
static_assert(kTileCount >= kWorkerWarps && kTileCount <= 2, "tcu_wmma_overlap supports up to 2 tiles");

static constexpr uint32_t required_payload_banks() {
  if constexpr (kWorkerWarps == 1) {
    return 2 * kABankSpan + 2 * kBBankSpan + 2 * kCBankSpan;
  } else {
    return kWorkerWarps * (kABankSpan + kBBankSpan + 2 * kCBankSpan);
  }
}

using a_tile_t = std::array<std::array<input_a_t, kAKDim>, kTileDim>;
using b_tile_t = std::array<std::array<input_b_t, kTileDim>, kBKDim>;
using c_tile_t = std::array<std::array<output_t, kTileDim>, kTileDim>;

vx_device_h device = nullptr;
vx_buffer_h input_a_buffer = nullptr;
vx_buffer_h input_meta_buffer = nullptr;
vx_buffer_h input_b_buffer = nullptr;
vx_buffer_h input_c_buffer = nullptr;
vx_buffer_h output_buffer = nullptr;
vx_buffer_h tma_desc_buffer = nullptr;
vx_buffer_h mma_desc_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static inline uint32_t a_desc_id(uint32_t tile_id, uint32_t phase) {
  return tile_id * kPhaseCount + phase;
}

static inline uint32_t b_desc_base() {
  return kTileCount * kPhaseCount;
}

static inline uint32_t b_desc_id(uint32_t tile_id, uint32_t phase) {
  return b_desc_base() + tile_id * kPhaseCount + phase;
}

static inline uint32_t c_in_desc_base() {
  return 2 * kTileCount * kPhaseCount;
}

static inline uint32_t c_in_desc_id(uint32_t tile_id) {
  return c_in_desc_base() + tile_id;
}

static inline uint32_t c_out_desc_base() {
  return c_in_desc_base() + kTileCount;
}

static inline uint32_t c_out_desc_id(uint32_t tile_id) {
  return c_out_desc_base() + tile_id;
}

void cleanup() {
  if (device) {
    vx_mem_free(input_a_buffer);
    vx_mem_free(input_meta_buffer);
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

template <typename T>
static inline void store_input_element(std::vector<uint8_t>& bytes, uint32_t byte_offset, T value) {
  if constexpr (std::is_same_v<T, uint16_t>) {
    bytes[byte_offset + 0] = value & 0xff;
    bytes[byte_offset + 1] = (value >> 8) & 0xff;
  } else {
    bytes[byte_offset] = value;
  }
}

static void pack_sparse_2_4_a_payload_and_meta(std::vector<uint8_t>& a_bytes,
                                               uint32_t a_byte_offset,
                                               std::vector<uint8_t>& meta_bytes,
                                               uint32_t meta_byte_offset,
                                               const a_tile_t& a_tile) {
  input_a_t compressed_tile[kTileDim][kACompressedK] = {};
  std::array<uint8_t, kMetaBytes> meta_tile = {};

  for (uint32_t step_m = 0; step_m < 2; ++step_m) {
    for (uint32_t step_k = 0; step_k < 2; ++step_k) {
      uint32_t line_index = step_m * 2 + step_k;
      for (uint32_t row = 0; row < 8; ++row) {
        uint32_t global_row = step_m * 8 + row;
        uint16_t row_meta = 0;
        for (uint32_t block = 0; block < 4; ++block) {
          uint32_t dense_col_base = step_k * 16 + block * 4;
          uint32_t sparse_col_base = step_k * 8 + block * 2;
          uint32_t picks[2] = {0, 0};
          input_a_t values[2] = {};
          uint32_t pick_count = 0;
          for (uint32_t lane = 0; lane < 4; ++lane) {
            auto value = a_tile[global_row][dense_col_base + lane];
            if (value != 0) {
              if (pick_count >= 2) {
                std::abort();
              }
              picks[pick_count] = lane;
              values[pick_count] = value;
              ++pick_count;
            }
          }
          if (pick_count != 2 || picks[0] == picks[1]) {
            std::abort();
          }
          compressed_tile[global_row][sparse_col_base + 0] = values[0];
          compressed_tile[global_row][sparse_col_base + 1] = values[1];
          row_meta |= static_cast<uint16_t>(picks[0] & 0x3) << (block * 4 + 0);
          row_meta |= static_cast<uint16_t>(picks[1] & 0x3) << (block * 4 + 2);
        }
        meta_tile[line_index * 16 + row * 2 + 0] = row_meta & 0xff;
        meta_tile[line_index * 16 + row * 2 + 1] = (row_meta >> 8) & 0xff;
      }
    }
  }

  host_utils::pack_ab_tile(a_bytes, a_byte_offset, compressed_tile, false);
  std::copy(meta_tile.begin(), meta_tile.end(), meta_bytes.begin() + meta_byte_offset);
}

static void pack_sparse_2_4_b_source(std::vector<uint8_t>& b_bytes,
                                     uint32_t byte_offset,
                                     const b_tile_t& b_tile) {
  constexpr uint32_t elem_bytes = sizeof(input_b_t);
  constexpr uint32_t line_elems = 16 * 8;
  constexpr uint32_t line_bytes = line_elems * elem_bytes;
  for (uint32_t step_n = 0; step_n < 2; ++step_n) {
    for (uint32_t step_k = 0; step_k < 2; ++step_k) {
      uint32_t line_index = step_n * 2 + step_k;
      uint32_t row_base = step_k * 16;
      uint32_t col_base = step_n * 8;
      uint32_t line_offset = byte_offset + line_index * line_bytes;
      for (uint32_t r = 0; r < 16; ++r) {
        for (uint32_t c = 0; c < 8; ++c) {
          uint32_t elem = r * 8 + c;
          store_input_element(b_bytes, line_offset + elem * elem_bytes, b_tile[row_base + r][col_base + c]);
        }
      }
    }
  }
}

static void pack_sparse_1_4_a_payload_and_meta(std::vector<uint8_t>& a_bytes,
                                               uint32_t a_byte_offset,
                                               std::vector<uint8_t>& meta_bytes,
                                               uint32_t meta_byte_offset,
                                               const a_tile_t& a_tile) {
  input_a_t compressed_tile[kTileDim][kACompressedK] = {};
  std::array<uint8_t, kMetaBytes> meta_tile = {};

  for (uint32_t step_m = 0; step_m < 2; ++step_m) {
    for (uint32_t step_k = 0; step_k < 2; ++step_k) {
      uint32_t line_index = step_m * 2 + step_k;
      for (uint32_t row = 0; row < 8; ++row) {
        uint32_t global_row = step_m * 8 + row;
        uint16_t row_meta = 0;
        for (uint32_t block = 0; block < 8; ++block) {
          uint32_t dense_col_base = step_k * 32 + block * 4;
          uint32_t sparse_col = step_k * 8 + block;
          uint32_t pick = 0;
          input_a_t value = {};
          uint32_t pick_count = 0;
          for (uint32_t lane = 0; lane < 4; ++lane) {
            auto lane_value = a_tile[global_row][dense_col_base + lane];
            if (lane_value != 0) {
              if (pick_count >= 1) {
                std::abort();
              }
              pick = lane;
              value = lane_value;
              ++pick_count;
            }
          }
          if (pick_count != 1) {
            std::abort();
          }
          compressed_tile[global_row][sparse_col] = value;
          row_meta |= static_cast<uint16_t>(pick & 0x3) << (block * 2);
        }
        meta_tile[line_index * 16 + row * 2 + 0] = row_meta & 0xff;
        meta_tile[line_index * 16 + row * 2 + 1] = (row_meta >> 8) & 0xff;
      }
    }
  }

  host_utils::pack_ab_tile(a_bytes, a_byte_offset, compressed_tile, false);
  std::copy(meta_tile.begin(), meta_tile.end(), meta_bytes.begin() + meta_byte_offset);
}

static void pack_sparse_1_4_b_source(std::vector<uint8_t>& b_bytes,
                                     uint32_t byte_offset,
                                     const b_tile_t& b_tile) {
  constexpr uint32_t elem_bytes = sizeof(input_b_t);
  constexpr uint32_t line_elems = 32 * 8;
  constexpr uint32_t line_bytes = line_elems * elem_bytes;
  for (uint32_t step_n = 0; step_n < 2; ++step_n) {
    for (uint32_t step_k = 0; step_k < 2; ++step_k) {
      uint32_t line_index = step_n * 2 + step_k;
      uint32_t row_base = step_k * 32;
      uint32_t col_base = step_n * 8;
      uint32_t line_offset = byte_offset + line_index * line_bytes;
      for (uint32_t r = 0; r < 32; ++r) {
        for (uint32_t c = 0; c < 8; ++c) {
          uint32_t elem = r * 8 + c;
          store_input_element(b_bytes, line_offset + elem * elem_bytes, b_tile[row_base + r][col_base + c]);
        }
      }
    }
  }
}

static void build_dense_chained_ref(float out[kTileDim][kTileDim],
                                    const std::array<a_tile_t, kPhaseCount>& a_tiles,
                                    const std::array<b_tile_t, kPhaseCount>& b_tiles,
                                    const c_tile_t& c_tile) {
  std::vector<uint8_t> c_bytes(kCBytes, 0);
  host_utils::pack_c_tile(c_bytes, 0, reinterpret_cast<const output_t (*)[kTileDim]>(c_tile.data()));

  std::vector<CMem::packet_t> c_packets(CMem::packet_count(host_utils::output_fmt()));
  for (size_t i = 0; i < c_packets.size(); ++i) {
    std::copy_n(c_bytes.data() + i * 64, 64, c_packets[i].begin());
  }

  CMem cmem;
  cmem.fill_tile(host_utils::output_fmt(), c_packets);

  std::array<std::array<std::array<uint32_t, 8>, 8>, 4> accum_fp22 = {};
  for (uint32_t subtile = 0; subtile < 4; ++subtile) {
    uint32_t storage_m = subtile / 2;
    uint32_t storage_n = subtile % 2;
    if constexpr (std::is_same_v<output_t, float>) {
      float c_block[8][8] = {};
      cmem.load_block_fp32(storage_m, storage_n, c_block);
      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
          accum_fp22.at(subtile).at(i).at(j) =
            convert_c_to_fp22(vortex::bit_cast<uint32_t>(c_block[i][j]), host_utils::out_precision());
        }
      }
    } else {
      uint16_t c_block[8][8] = {};
      cmem.load_block_fp16(storage_m, storage_n, c_block);
      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
          accum_fp22.at(subtile).at(i).at(j) = convert_c_to_fp22(c_block[i][j], host_utils::out_precision());
        }
      }
    }
  }

  for (uint32_t phase = 0; phase < kPhaseCount; ++phase) {
    std::vector<uint8_t> a_bytes(kABytes, 0);
    std::vector<uint8_t> b_bytes(kBBytes, 0);
    input_a_t a_dense[kTileDim][kTileDim] = {};
    input_b_t b_dense[kTileDim][kTileDim] = {};
    for (uint32_t i = 0; i < kTileDim; ++i) {
      for (uint32_t j = 0; j < kTileDim; ++j) {
        a_dense[i][j] = a_tiles[phase][i][j];
        b_dense[i][j] = b_tiles[phase][i][j];
      }
    }
    host_utils::pack_ab_tile(a_bytes, 0, a_dense, false);
    host_utils::pack_ab_tile(b_bytes, 0, b_dense, true);

    std::vector<AMem::packet_t> a_packets(AMem::packet_count(host_utils::input_fmt<input_a_t>()));
    std::vector<BMem::packet_t> b_packets(BMem::packet_count(host_utils::input_fmt<input_b_t>()));
    for (size_t i = 0; i < a_packets.size(); ++i) {
      std::copy_n(a_bytes.data() + i * 64, 64, a_packets[i].begin());
    }
    for (size_t i = 0; i < b_packets.size(); ++i) {
      std::copy_n(b_bytes.data() + i * 64, 64, b_packets[i].begin());
    }

    AMem amem;
    BMem bmem;
    amem.fill_tile(host_utils::input_fmt<input_a_t>(), a_packets);
    bmem.fill_tile(host_utils::input_fmt<input_b_t>(), b_packets);

    for (uint32_t storage_k = 0; storage_k < 2; ++storage_k) {
      for (uint32_t storage_m = 0; storage_m < 2; ++storage_m) {
        for (uint32_t storage_n = 0; storage_n < 2; ++storage_n) {
          uint16_t a_block[8][8] = {};
          uint16_t b_block[8][8] = {};
          uint32_t partial_fp22[8][8] = {};
          uint32_t subtile = storage_m * 2 + storage_n;
          amem.read_primitive(storage_m, storage_k, a_block);
          bmem.read_primitive(storage_k, storage_n, b_block);
          host_utils::run_open_tensorcore_primitive_fp22(a_block, b_block, partial_fp22);
          for (uint32_t i = 0; i < 8; ++i) {
            for (uint32_t j = 0; j < 8; ++j) {
              accum_fp22.at(subtile).at(i).at(j) =
                host_utils::add_fp22_raw(accum_fp22.at(subtile).at(i).at(j), partial_fp22[i][j]);
            }
          }
        }
      }
    }
  }

  for (uint32_t subtile = 0; subtile < 4; ++subtile) {
    uint32_t storage_m = subtile / 2;
    uint32_t storage_n = subtile % 2;
    if constexpr (std::is_same_v<output_t, float>) {
      float c_block[8][8] = {};
      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
          c_block[i][j] = vortex::bit_cast<float>(fp22_to_fp32(accum_fp22.at(subtile).at(i).at(j)));
        }
      }
      cmem.store_block_fp32(storage_m, storage_n, c_block);
    } else {
      uint16_t c_block[8][8] = {};
      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
          c_block[i][j] = fp22_to_fp16(accum_fp22.at(subtile).at(i).at(j));
        }
      }
      cmem.store_block_fp16(storage_m, storage_n, c_block);
    }
  }

  std::vector<CMem::packet_t> out_packets;
  cmem.dump_tile(host_utils::output_fmt(), &out_packets);
  std::vector<uint8_t> tile_bytes(kCBytes, 0);
  for (size_t i = 0; i < out_packets.size(); ++i) {
    std::copy_n(out_packets[i].begin(), 64, tile_bytes.begin() + i * 64);
  }

  std::vector<output_t> accum(kTileDim * kTileDim, host_utils::encode_output(0.0f));
  host_utils::scatter_c_tile(accum, tile_bytes.data(), 0, 0);
  for (uint32_t i = 0; i < kTileDim; ++i) {
    for (uint32_t j = 0; j < kTileDim; ++j) {
      out[i][j] = host_utils::decode_output(accum[i * kTileDim + j]);
    }
  }
}

static void build_sparse_chained_ref(float out[kTileDim][kTileDim],
                                     const std::array<a_tile_t, kPhaseCount>& a_tiles,
                                     const std::array<b_tile_t, kPhaseCount>& b_tiles,
                                     const c_tile_t& c_tile) {
  std::vector<uint8_t> c_bytes(kCBytes, 0);
  host_utils::pack_c_tile(c_bytes, 0, reinterpret_cast<const output_t (*)[kTileDim]>(c_tile.data()));

  std::vector<CMem::packet_t> c_packets(CMem::packet_count(host_utils::output_fmt()));
  for (size_t i = 0; i < c_packets.size(); ++i) {
    std::copy_n(c_bytes.data() + i * 64, 64, c_packets[i].begin());
  }

  CMem cmem;
  cmem.fill_tile(host_utils::output_fmt(), c_packets);

  std::array<std::array<std::array<uint32_t, 8>, 8>, 4> accum_fp22 = {};
  for (uint32_t subtile = 0; subtile < 4; ++subtile) {
    uint32_t storage_m = subtile / 2;
    uint32_t storage_n = subtile % 2;
    if constexpr (std::is_same_v<output_t, float>) {
      float c_block[8][8] = {};
      cmem.load_block_fp32(storage_m, storage_n, c_block);
      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
          accum_fp22.at(subtile).at(i).at(j) =
            convert_c_to_fp22(vortex::bit_cast<uint32_t>(c_block[i][j]), host_utils::out_precision());
        }
      }
    } else {
      uint16_t c_block[8][8] = {};
      cmem.load_block_fp16(storage_m, storage_n, c_block);
      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
          accum_fp22.at(subtile).at(i).at(j) = convert_c_to_fp22(c_block[i][j], host_utils::out_precision());
        }
      }
    }
  }

  for (uint32_t phase = 0; phase < kPhaseCount; ++phase) {
    std::vector<uint8_t> a_bytes(kABytes, 0);
    std::vector<uint8_t> meta_bytes(kMetaBytes, 0);
    std::vector<uint8_t> b_bytes(kBBytes, 0);
    if constexpr (kSparse2To4) {
      pack_sparse_2_4_a_payload_and_meta(a_bytes, 0, meta_bytes, 0, a_tiles[phase]);
      pack_sparse_2_4_b_source(b_bytes, 0, b_tiles[phase]);
    } else {
      pack_sparse_1_4_a_payload_and_meta(a_bytes, 0, meta_bytes, 0, a_tiles[phase]);
      pack_sparse_1_4_b_source(b_bytes, 0, b_tiles[phase]);
    }

    std::vector<AMem::packet_t> a_packets(AMem::packet_count(host_utils::input_fmt<input_a_t>()));
    std::vector<MetaMem::packet_t> meta_packets(MetaMem::packet_count());
    std::vector<BMem::packet_t> b_packets(BMem::packet_count(host_utils::input_fmt<input_b_t>(),
                                                             kSparse2To4 ? vt::sparse_2_4 : vt::sparse_1_4));
    for (size_t i = 0; i < a_packets.size(); ++i) {
      std::copy_n(a_bytes.data() + i * 64, 64, a_packets[i].begin());
    }
    for (size_t i = 0; i < meta_packets.size(); ++i) {
      std::copy_n(meta_bytes.data() + i * 64, 64, meta_packets[i].begin());
    }
    for (size_t i = 0; i < b_packets.size(); ++i) {
      std::copy_n(b_bytes.data() + i * 64, 64, b_packets[i].begin());
    }

    AMem amem;
    MetaMem metamem;
    BMem bmem;
    amem.fill_tile(host_utils::input_fmt<input_a_t>(), a_packets);
    metamem.fill_tile(meta_packets);
    bmem.fill_tile(host_utils::input_fmt<input_b_t>(), b_packets, kSparse2To4 ? vt::sparse_2_4 : vt::sparse_1_4);

    for (uint32_t storage_k = 0; storage_k < 2; ++storage_k) {
      for (uint32_t storage_m = 0; storage_m < 2; ++storage_m) {
        for (uint32_t storage_n = 0; storage_n < 2; ++storage_n) {
          uint16_t a_block[8][8] = {};
          uint16_t b_block[8][8] = {};
          uint8_t meta_line[MetaMem::kLineBytes] = {};
          uint32_t partial_fp22[8][8] = {};
          uint32_t subtile = storage_m * 2 + storage_n;
          amem.read_primitive(storage_m, storage_k, a_block);
          metamem.read_line(storage_m, storage_k, meta_line);
          if constexpr (kSparse2To4) {
            uint16_t b_source[16][8] = {};
            bmem.read_sparse_2_4_source(storage_k, storage_n, b_source);
            sparse_select_2_4(meta_line, b_source, b_block);
          } else {
            uint16_t b_source[32][8] = {};
            bmem.read_sparse_1_4_source(storage_k, storage_n, b_source);
            sparse_select_1_4(meta_line, b_source, b_block);
          }
          host_utils::run_open_tensorcore_primitive_fp22(a_block, b_block, partial_fp22);
          for (uint32_t i = 0; i < 8; ++i) {
            for (uint32_t j = 0; j < 8; ++j) {
              accum_fp22.at(subtile).at(i).at(j) =
                host_utils::add_fp22_raw(accum_fp22.at(subtile).at(i).at(j), partial_fp22[i][j]);
            }
          }
        }
      }
    }
  }

  for (uint32_t subtile = 0; subtile < 4; ++subtile) {
    uint32_t storage_m = subtile / 2;
    uint32_t storage_n = subtile % 2;
    if constexpr (std::is_same_v<output_t, float>) {
      float c_block[8][8] = {};
      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
          c_block[i][j] = vortex::bit_cast<float>(fp22_to_fp32(accum_fp22.at(subtile).at(i).at(j)));
        }
      }
      cmem.store_block_fp32(storage_m, storage_n, c_block);
    } else {
      uint16_t c_block[8][8] = {};
      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
          c_block[i][j] = fp22_to_fp16(accum_fp22.at(subtile).at(i).at(j));
        }
      }
      cmem.store_block_fp16(storage_m, storage_n, c_block);
    }
  }

  std::vector<CMem::packet_t> out_packets;
  cmem.dump_tile(host_utils::output_fmt(), &out_packets);
  std::vector<uint8_t> tile_bytes(kCBytes, 0);
  for (size_t i = 0; i < out_packets.size(); ++i) {
    std::copy_n(out_packets[i].begin(), 64, tile_bytes.begin() + i * 64);
  }

  std::vector<output_t> accum(kTileDim * kTileDim, host_utils::encode_output(0.0f));
  host_utils::scatter_c_tile(accum, tile_bytes.data(), 0, 0);
  for (uint32_t i = 0; i < kTileDim; ++i) {
    for (uint32_t j = 0; j < kTileDim; ++j) {
      out[i][j] = host_utils::decode_output(accum[i * kTileDim + j]);
    }
  }
}

static void build_chained_ref(float out[kTileDim][kTileDim],
                              const std::array<a_tile_t, kPhaseCount>& a_tiles,
                              const std::array<b_tile_t, kPhaseCount>& b_tiles,
                              const c_tile_t& c_tile) {
  if constexpr (kHasSparse) {
    build_sparse_chained_ref(out, a_tiles, b_tiles, c_tile);
  } else {
    build_dense_chained_ref(out, a_tiles, b_tiles, c_tile);
  }
}

int main() {
  if constexpr (required_payload_banks() > kTmemPayloadBanks) {
    std::cout << "Unsupported TMEM payload footprint: need "
              << required_payload_banks() << " payload banks, but simx provides "
              << kTmemPayloadBanks << "." << std::endl;
    return -1;
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

  std::array<std::array<a_tile_t, kPhaseCount>, kTileCount> a_tiles = {};
  std::array<std::array<b_tile_t, kPhaseCount>, kTileCount> b_tiles = {};
  std::array<c_tile_t, kTileCount> c_tiles = {};
  std::array<std::array<std::array<float, kTileDim>, kTileDim>, kTileCount> ref_tiles = {};

  for (uint32_t tile_id = 0; tile_id < kTileCount; ++tile_id) {
    for (uint32_t phase = 0; phase < kPhaseCount; ++phase) {
      for (uint32_t i = 0; i < kTileDim; ++i) {
        if constexpr (kHasSparse) {
          for (uint32_t j = 0; j < kAKDim; ++j) {
            a_tiles[tile_id][phase][i][j] = host_utils::encode_a_input(0.0f);
          }
          for (uint32_t block = 0; block < (kAKDim / 4); ++block) {
            uint32_t dense_col_base = block * 4;
            if constexpr (kSparse2To4) {
              uint32_t sel0 = (tile_id + phase + i + block) & 0x3;
              uint32_t sel1 = (sel0 + 2) & 0x3;
              uint32_t col0 = dense_col_base + sel0;
              uint32_t col1 = dense_col_base + sel1;
              float a_val0 = 0.0625f * float((tile_id + 1) * 11 + (phase + 1) * 3 + i + block + 1);
              float a_val1 = 0.0625f * float((tile_id + 1) * 13 + (phase + 1) * 5 + i + block + 2);
              a_tiles[tile_id][phase][i][col0] = host_utils::encode_a_input(a_val0);
              a_tiles[tile_id][phase][i][col1] = host_utils::encode_a_input(a_val1);
            } else {
              uint32_t sel = (tile_id + phase + i + block) & 0x3;
              uint32_t col = dense_col_base + sel;
              float a_val = 0.0625f * float((tile_id + 1) * 17 + (phase + 1) * 7 + i + block + 1);
              a_tiles[tile_id][phase][i][col] = host_utils::encode_a_input(a_val);
            }
          }
        } else {
          for (uint32_t j = 0; j < kAKDim; ++j) {
            float a_val = 0.0625f * float((tile_id + 1) * 7 + (phase + 1) * 3 + i + (j % 5));
            a_tiles[tile_id][phase][i][j] = host_utils::encode_a_input(a_val);
          }
        }
      }
      for (uint32_t i = 0; i < kBKDim; ++i) {
        for (uint32_t j = 0; j < kTileDim; ++j) {
          float b_val = 0.03125f * float((tile_id + 1) * 9 + (phase + 1) * 5 + j + (i % 11));
          b_tiles[tile_id][phase][i][j] = host_utils::encode_b_input(b_val);
        }
      }
    }
    for (uint32_t i = 0; i < kTileDim; ++i) {
      for (uint32_t j = 0; j < kTileDim; ++j) {
        float c_val = 0.125f * float(i == j) + 0.03125f * float(tile_id);
        c_tiles[tile_id][i][j] = host_utils::encode_output(c_val);
      }
    }
    build_chained_ref(reinterpret_cast<float (*)[kTileDim]>(ref_tiles[tile_id].data()),
                      a_tiles[tile_id],
                      b_tiles[tile_id],
                      c_tiles[tile_id]);
  }

  std::vector<uint8_t> h_a(kTileCount * kPhaseCount * kABytes, 0);
  std::vector<uint8_t> h_meta(kTileCount * kPhaseCount * kMetaBytes, 0);
  std::vector<uint8_t> h_b(kTileCount * kPhaseCount * kBBytes, 0);
  std::vector<uint8_t> h_c(kTileCount * kCBytes, 0);

  for (uint32_t tile_id = 0; tile_id < kTileCount; ++tile_id) {
    for (uint32_t phase = 0; phase < kPhaseCount; ++phase) {
      uint32_t a_offset = (tile_id * kPhaseCount + phase) * kABytes;
      uint32_t meta_offset = (tile_id * kPhaseCount + phase) * kMetaBytes;
      uint32_t b_offset = (tile_id * kPhaseCount + phase) * kBBytes;
      if constexpr (kHasSparse) {
        if constexpr (kSparse2To4) {
          pack_sparse_2_4_a_payload_and_meta(h_a, a_offset, h_meta, meta_offset, a_tiles[tile_id][phase]);
          pack_sparse_2_4_b_source(h_b, b_offset, b_tiles[tile_id][phase]);
        } else {
          pack_sparse_1_4_a_payload_and_meta(h_a, a_offset, h_meta, meta_offset, a_tiles[tile_id][phase]);
          pack_sparse_1_4_b_source(h_b, b_offset, b_tiles[tile_id][phase]);
        }
      } else {
        input_a_t a_dense[kTileDim][kTileDim] = {};
        input_b_t b_dense[kTileDim][kTileDim] = {};
        for (uint32_t i = 0; i < kTileDim; ++i) {
          for (uint32_t j = 0; j < kTileDim; ++j) {
            a_dense[i][j] = a_tiles[tile_id][phase][i][j];
            b_dense[i][j] = b_tiles[tile_id][phase][i][j];
          }
        }
        host_utils::pack_ab_tile(h_a, a_offset, a_dense, false);
        host_utils::pack_ab_tile(h_b, b_offset, b_dense, true);
      }
    }
    host_utils::pack_c_tile(h_c, tile_id * kCBytes, reinterpret_cast<const output_t (*)[kTileDim]>(c_tiles[tile_id].data()));
  }

  std::vector<tma_descriptor_t> tma_descs(c_out_desc_id(kTileCount - 1) + 1);
  std::vector<mma_descriptor_t> mma_descs(1);

  RT_CHECK(vx_mem_alloc(device, h_a.size(), VX_MEM_READ, &input_a_buffer));
  if constexpr (kHasSparse) {
    RT_CHECK(vx_mem_alloc(device, h_meta.size(), VX_MEM_READ, &input_meta_buffer));
  }
  RT_CHECK(vx_mem_alloc(device, h_b.size(), VX_MEM_READ, &input_b_buffer));
  RT_CHECK(vx_mem_alloc(device, h_c.size(), VX_MEM_READ, &input_c_buffer));
  RT_CHECK(vx_mem_alloc(device, kTileCount * kCBytes, VX_MEM_WRITE, &output_buffer));
  RT_CHECK(vx_mem_alloc(device, tma_descs.size() * sizeof(tma_descriptor_t), VX_MEM_READ, &tma_desc_buffer));
  RT_CHECK(vx_mem_alloc(device, mma_descs.size() * sizeof(mma_descriptor_t), VX_MEM_READ, &mma_desc_buffer));

  uint64_t input_a_addr = 0;
  uint64_t input_meta_addr = 0;
  uint64_t input_b_addr = 0;
  uint64_t input_c_addr = 0;
  uint64_t output_addr = 0;
  uint64_t tma_desc_table_addr = 0;
  uint64_t mma_desc_table_addr = 0;
  RT_CHECK(vx_mem_address(input_a_buffer, &input_a_addr));
  if constexpr (kHasSparse) {
    RT_CHECK(vx_mem_address(input_meta_buffer, &input_meta_addr));
  }
  RT_CHECK(vx_mem_address(input_b_buffer, &input_b_addr));
  RT_CHECK(vx_mem_address(input_c_buffer, &input_c_addr));
  RT_CHECK(vx_mem_address(output_buffer, &output_addr));

  for (uint32_t tile_id = 0; tile_id < kTileCount; ++tile_id) {
    for (uint32_t phase = 0; phase < kPhaseCount; ++phase) {
      uint32_t adesc = a_desc_id(tile_id, phase);
      uint32_t bdesc = b_desc_id(tile_id, phase);
      uint32_t phase_index = tile_id * kPhaseCount + phase;
      tma_descs[adesc] = {};
      tma_descs[adesc].addr = input_a_addr + phase_index * kABytes;
      tma_descs[adesc].size_bytes = kABytes;
      tma_descs[adesc].tile_role = kTileRoleA;
      tma_descs[adesc].payload_kind = kHasSparse ? kPayloadSparsePayload : kPayloadDense;
      if constexpr (kHasSparse) {
        tma_descs[adesc].meta_addr = input_meta_addr + phase_index * kMetaBytes;
        tma_descs[adesc].meta_size_bytes = kMetaBytes;
        tma_descs[adesc].meta_tmem_base = kMetaBankBase + (phase & 0x1);
        tma_descs[adesc].meta_col_span = kMetaBankSpan;
      }

      tma_descs[bdesc] = {};
      tma_descs[bdesc].addr = input_b_addr + phase_index * kBBytes;
      tma_descs[bdesc].size_bytes = kBBytes;
      tma_descs[bdesc].tile_role = kTileRoleB;
      tma_descs[bdesc].payload_kind = kPayloadDense;
    }

    tma_descs[c_in_desc_id(tile_id)] = {};
    tma_descs[c_in_desc_id(tile_id)].addr = input_c_addr + tile_id * kCBytes;
    tma_descs[c_in_desc_id(tile_id)].size_bytes = kCBytes;
    tma_descs[c_in_desc_id(tile_id)].tile_role = kTileRoleC;
    tma_descs[c_in_desc_id(tile_id)].payload_kind = kPayloadDense;

    tma_descs[c_out_desc_id(tile_id)] = {};
    tma_descs[c_out_desc_id(tile_id)].addr = output_addr + tile_id * kCBytes;
    tma_descs[c_out_desc_id(tile_id)].size_bytes = kCBytes;
    tma_descs[c_out_desc_id(tile_id)].tile_role = kTileRoleC;
    tma_descs[c_out_desc_id(tile_id)].payload_kind = kPayloadDense;
  }

  mma_descs[0].fmt_a = vt::ATYPE::id;
  mma_descs[0].fmt_b = vt::BTYPE::id;
  mma_descs[0].fmt_c = vt::OTYPE::id;
  mma_descs[0].fmt_d = vt::OTYPE::id;
  mma_descs[0].sparse_mode = kSparse2To4 ? vt::sparse_2_4 : (kSparse1To4 ? vt::sparse_1_4 : vt::sparse_none);
  mma_descs[0].a_rows = kTileDim;
  mma_descs[0].a_cols = kAKDim;
  mma_descs[0].b_rows = kBKDim;
  mma_descs[0].b_cols = kTileDim;
  mma_descs[0].c_rows = kTileDim;
  mma_descs[0].c_cols = kTileDim;

  RT_CHECK(vx_copy_to_dev(input_a_buffer, h_a.data(), 0, h_a.size()));
  if constexpr (kHasSparse) {
    RT_CHECK(vx_copy_to_dev(input_meta_buffer, h_meta.data(), 0, h_meta.size()));
  }
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
  kernel_arg.meta_col_span = kMetaBankSpan;

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  std::vector<uint8_t> h_output_bytes(kTileCount * kCBytes, 0);
  RT_CHECK(vx_copy_from_dev(h_output_bytes.data(), output_buffer, 0, h_output_bytes.size()));

  float max_abs_err = 0.0f;
  int errors = 0;
  constexpr float tolerance = 1e-6f;
  for (uint32_t tile_id = 0; tile_id < kTileCount; ++tile_id) {
    std::vector<output_t> h_output(kTileDim * kTileDim, host_utils::encode_output(0.0f));
    host_utils::scatter_c_tile(h_output, h_output_bytes.data() + tile_id * kCBytes, 0, 0);
    std::vector<float> h_output_float;
    host_utils::convert_output_matrix_to_float(h_output_float, h_output);
    for (uint32_t i = 0; i < kTileDim; ++i) {
      for (uint32_t j = 0; j < kTileDim; ++j) {
        uint32_t idx = i * kTileDim + j;
        auto actual = h_output_float[idx];
        auto expect = ref_tiles[tile_id][i][j];
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
