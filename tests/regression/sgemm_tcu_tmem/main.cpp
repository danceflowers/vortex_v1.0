#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <tensor_cfg.h>
#include <util.h>
#include <vortex.h>

#include "../tcu_host_utils.h"
#include "common.h"
#include "open_tensorcore/meta_mem.h"
#include "open_tensorcore/sparse_select.h"

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

static constexpr bool kSparse2To4 = (SPARSE_MODE == 1);
static constexpr bool kSparse1To4 = (SPARSE_MODE == 2);
static constexpr bool kHasSparse = kSparse2To4 || kSparse1To4;
static_assert(SPARSE_MODE == 0 || SPARSE_MODE == 1 || SPARSE_MODE == 2,
              "sgemm_tcu_tmem currently supports only dense, 2:4 sparse, or 1:4 sparse");
static constexpr uint32_t kMatrixM = 128;
static constexpr uint32_t kMatrixN = 128;
static constexpr uint32_t kMatrixK = 128;
static constexpr uint32_t kTileDim = 16;
static constexpr uint32_t kKSlice = kSparse1To4 ? 64 : (kSparse2To4 ? 32 : 16);
static constexpr uint32_t kTileRows = kMatrixM / kTileDim;
static constexpr uint32_t kTileCols = kMatrixN / kTileDim;
static constexpr uint32_t kKPhases = kMatrixK / kKSlice;
static constexpr uint32_t kNumTiles = kTileRows * kTileCols;
static constexpr uint32_t kACompressedK = 16;
static constexpr uint32_t kABytes = kTileDim * kACompressedK * sizeof(input_a_t);
static constexpr uint32_t kMetaBytes = kHasSparse ? 64 : 0;
static constexpr uint32_t kBBytes = kKSlice * kTileDim * sizeof(input_b_t);
static constexpr uint32_t kCBytes = kTileDim * kTileDim * sizeof(output_t);
static constexpr uint32_t ceil_div(uint32_t a, uint32_t b) {
  return (a + b - 1) / b;
}
static constexpr uint32_t kABankSpan = kKSlice * sizeof(input_a_t);
static constexpr uint32_t kBBankSpan = kTileDim * sizeof(input_b_t);
static constexpr uint32_t kCBankSpan = kTileDim * sizeof(output_t);
static constexpr uint32_t kMetaBankBase = 16;
static constexpr uint32_t kMetaBankSpan = kHasSparse ? 1 : 0;
static constexpr uint8_t kTileRoleA = 1;
static constexpr uint8_t kTileRoleB = 2;
static constexpr uint8_t kTileRoleC = 3;
static constexpr uint8_t kPayloadDense = 0;
static constexpr uint8_t kPayloadSparsePayload = 1;
using host_utils = tcu_test::TileHostUtils<input_a_t, input_b_t, output_t, kTileDim>;

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

static void gather_sparse_a_phase_tile(input_a_t tile[kTileDim][kKSlice],
                                       const std::vector<input_a_t>& matrix,
                                       uint32_t tile_row,
                                       uint32_t k_phase) {
  uint32_t row_base = tile_row * kTileDim;
  uint32_t col_base = k_phase * kKSlice;
  for (uint32_t i = 0; i < kTileDim; ++i) {
    for (uint32_t j = 0; j < kKSlice; ++j) {
      tile[i][j] = matrix[(row_base + i) * kMatrixK + (col_base + j)];
    }
  }
}

static void gather_sparse_b_phase_tile(input_b_t tile[kKSlice][kTileDim],
                                       const std::vector<input_b_t>& matrix,
                                       uint32_t k_phase,
                                       uint32_t tile_col) {
  uint32_t row_base = k_phase * kKSlice;
  uint32_t col_base = tile_col * kTileDim;
  for (uint32_t i = 0; i < kKSlice; ++i) {
    for (uint32_t j = 0; j < kTileDim; ++j) {
      tile[i][j] = matrix[(row_base + i) * kMatrixN + (col_base + j)];
    }
  }
}

static void pack_sparse_2_4_a_payload_and_meta(std::vector<uint8_t>& a_bytes,
                                               uint32_t a_byte_offset,
                                               std::vector<uint8_t>& meta_bytes,
                                               uint32_t meta_byte_offset,
                                               const input_a_t a_tile[kTileDim][kKSlice]) {
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
                                     const input_b_t b_tile[kKSlice][kTileDim]) {
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
                                               const input_a_t a_tile[kTileDim][kKSlice]) {
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
                                     const input_b_t b_tile[kKSlice][kTileDim]) {
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

static void build_all_phase_input(std::vector<uint8_t>& a_bytes,
                                  std::vector<uint8_t>& meta_bytes,
                                  std::vector<uint8_t>& b_bytes,
                                  const std::vector<input_a_t>& a_matrix,
                                  const std::vector<input_b_t>& b_matrix,
                                  uint32_t phase_limit) {
  a_bytes.assign(phase_limit * kNumTiles * kABytes, 0);
  meta_bytes.assign(kHasSparse ? (phase_limit * kNumTiles * kMetaBytes) : 0, 0);
  b_bytes.assign(phase_limit * kNumTiles * kBBytes, 0);
  for (uint32_t phase = 0; phase < phase_limit; ++phase) {
    for (uint32_t tile_row = 0; tile_row < kTileRows; ++tile_row) {
      for (uint32_t tile_col = 0; tile_col < kTileCols; ++tile_col) {
        uint32_t tile_id = tile_row * kTileCols + tile_col;
        uint32_t phase_tile_id = phase * kNumTiles + tile_id;
        if constexpr (kHasSparse) {
          input_a_t a_tile[kTileDim][kKSlice];
          input_b_t b_tile[kKSlice][kTileDim];
          gather_sparse_a_phase_tile(a_tile, a_matrix, tile_row, phase);
          gather_sparse_b_phase_tile(b_tile, b_matrix, phase, tile_col);
          if constexpr (kSparse2To4) {
            pack_sparse_2_4_a_payload_and_meta(a_bytes,
                                               phase_tile_id * kABytes,
                                               meta_bytes,
                                               phase_tile_id * kMetaBytes,
                                               a_tile);
            pack_sparse_2_4_b_source(b_bytes, phase_tile_id * kBBytes, b_tile);
          } else {
            pack_sparse_1_4_a_payload_and_meta(a_bytes,
                                               phase_tile_id * kABytes,
                                               meta_bytes,
                                               phase_tile_id * kMetaBytes,
                                               a_tile);
            pack_sparse_1_4_b_source(b_bytes, phase_tile_id * kBBytes, b_tile);
          }
        } else {
          input_a_t a_tile[kTileDim][kTileDim];
          input_b_t b_tile[kTileDim][kTileDim];
          host_utils::gather_a_phase_tile(a_tile, a_matrix, kMatrixK, tile_row, phase, kKSlice);
          host_utils::gather_b_phase_tile(b_tile, b_matrix, kMatrixN, phase, tile_col, kKSlice);
          host_utils::pack_ab_tile(a_bytes, phase_tile_id * kABytes, a_tile, false);
          host_utils::pack_ab_tile(b_bytes, phase_tile_id * kBBytes, b_tile, true);
        }
      }
    }
  }
}

static void build_chained_tile_ref(float out[kTileDim][kTileDim],
                                   const std::vector<input_a_t>& a_matrix,
                                   const std::vector<input_b_t>& b_matrix,
                                   uint32_t tile_row,
                                   uint32_t tile_col,
                                   uint32_t phase_limit) {
  std::vector<uint8_t> c_bytes(kCBytes, 0);
  output_t c_tile[kTileDim][kTileDim] = {};
  host_utils::pack_c_tile(c_bytes, 0, c_tile);

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

  for (uint32_t phase = 0; phase < phase_limit; ++phase) {
    std::vector<uint8_t> a_bytes(kABytes, 0);
    std::vector<uint8_t> b_bytes(kBBytes, 0);

    if constexpr (kHasSparse) {
      input_a_t a_tile[kTileDim][kKSlice];
      input_b_t b_tile[kKSlice][kTileDim];
      std::vector<uint8_t> meta_bytes(kMetaBytes, 0);
      gather_sparse_a_phase_tile(a_tile, a_matrix, tile_row, phase);
      gather_sparse_b_phase_tile(b_tile, b_matrix, phase, tile_col);
      if constexpr (kSparse2To4) {
        pack_sparse_2_4_a_payload_and_meta(a_bytes, 0, meta_bytes, 0, a_tile);
        pack_sparse_2_4_b_source(b_bytes, 0, b_tile);
      } else {
        pack_sparse_1_4_a_payload_and_meta(a_bytes, 0, meta_bytes, 0, a_tile);
        pack_sparse_1_4_b_source(b_bytes, 0, b_tile);
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
    } else {
      input_a_t a_tile[kTileDim][kTileDim];
      input_b_t b_tile[kTileDim][kTileDim];
      host_utils::gather_a_phase_tile(a_tile, a_matrix, kMatrixK, tile_row, phase, kKSlice);
      host_utils::gather_b_phase_tile(b_tile, b_matrix, kMatrixN, phase, tile_col, kKSlice);
      host_utils::pack_ab_tile(a_bytes, 0, a_tile, false);
      host_utils::pack_ab_tile(b_bytes, 0, b_tile, true);

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

static void build_open_tensorcore_ref(std::vector<float>& out,
                                      const std::vector<input_a_t>& a_matrix,
                                      const std::vector<input_b_t>& b_matrix,
                                      uint32_t phase_limit) {
  std::vector<output_t> accum(kMatrixM * kMatrixN, host_utils::encode_output(0.0f));
  for (uint32_t tile_row = 0; tile_row < kTileRows; ++tile_row) {
    for (uint32_t tile_col = 0; tile_col < kTileCols; ++tile_col) {
      float ref_tile[kTileDim][kTileDim];
      build_chained_tile_ref(ref_tile, a_matrix, b_matrix, tile_row, tile_col, phase_limit);
      host_utils::store_output_tile(accum, ref_tile, kMatrixN, tile_row, tile_col);
    }
  }
  host_utils::convert_output_matrix_to_float(out, accum);
}

static void build_fp32_ref(std::vector<float>& out,
                           const std::vector<input_a_t>& a_matrix,
                           const std::vector<input_b_t>& b_matrix,
                           uint32_t phase_limit) {
  out.assign(kMatrixM * kMatrixN, 0.0f);
  for (uint32_t m = 0; m < kMatrixM; ++m) {
    for (uint32_t n = 0; n < kMatrixN; ++n) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < phase_limit * kKSlice; ++k) {
        float a = host_utils::decode_a_input(a_matrix[m * kMatrixK + k]);
        float b = host_utils::decode_b_input(b_matrix[k * kMatrixN + n]);
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
      if constexpr (kHasSparse) {
        uint32_t lane = k & 0x3;
        if constexpr (kSparse2To4) {
          uint32_t pick0 = (i + (k >> 2)) & 0x3;
          uint32_t pick1 = (pick0 + 2) & 0x3;
          if (lane == pick0 || lane == pick1) {
            if (value == 0.0f) {
              value = 0.0625f;
            }
            h_a[i * kMatrixK + k] = host_utils::encode_a_input(value);
          } else {
            h_a[i * kMatrixK + k] = host_utils::encode_a_input(0.0f);
          }
        } else {
          uint32_t pick = (i + (k >> 2)) & 0x3;
          if (lane == pick) {
            if (value == 0.0f) {
              value = 0.0625f;
            }
            h_a[i * kMatrixK + k] = host_utils::encode_a_input(value);
          } else {
            h_a[i * kMatrixK + k] = host_utils::encode_a_input(0.0f);
          }
        }
      } else {
        h_a[i * kMatrixK + k] = host_utils::encode_a_input(value);
      }
    }
  }
  for (uint32_t k = 0; k < kMatrixK; ++k) {
    for (uint32_t j = 0; j < kMatrixN; ++j) {
      int32_t raw = static_cast<int32_t>(((k * 11) + (j * 5)) % 37) - 18;
      float value = static_cast<float>(raw) * 0.0625f;
      h_b[k * kMatrixN + j] = host_utils::encode_b_input(value);
    }
  }

  std::vector<float> h_quant_ref;
  std::vector<float> h_fp32_ref;
  std::cout << "build OpenTensorCore reference" << std::endl;
  build_open_tensorcore_ref(h_quant_ref, h_a, h_b, phase_limit);
  std::cout << "build FP32 accumulate reference" << std::endl;
  build_fp32_ref(h_fp32_ref, h_a, h_b, phase_limit);

  std::vector<uint8_t> h_all_phase_a;
  std::vector<uint8_t> h_all_phase_meta;
  std::vector<uint8_t> h_all_phase_b;
  std::vector<uint8_t> h_init_c(kNumTiles * kCBytes, 0);
  std::vector<uint8_t> h_output_tiles(kNumTiles * kCBytes, 0);

  build_all_phase_input(h_all_phase_a, h_all_phase_meta, h_all_phase_b, h_a, h_b, phase_limit);

  uint32_t a_desc_count = phase_limit * kNumTiles;
  uint32_t b_desc_count = phase_limit * kNumTiles;
  uint32_t c_desc_count = kNumTiles;
  std::vector<tma_descriptor_t> tma_descs(a_desc_count + b_desc_count + 2 * c_desc_count);
  std::vector<mma_descriptor_t> mma_descs(1);

  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, h_all_phase_a.size(), VX_MEM_READ, &input_a_buffer));
  if constexpr (kHasSparse) {
    RT_CHECK(vx_mem_alloc(device, h_all_phase_meta.size(), VX_MEM_READ, &input_meta_buffer));
  }
  RT_CHECK(vx_mem_alloc(device, h_all_phase_b.size(), VX_MEM_READ, &input_b_buffer));
  RT_CHECK(vx_mem_alloc(device, h_init_c.size(), VX_MEM_READ, &input_c_buffer));
  RT_CHECK(vx_mem_alloc(device, h_output_tiles.size(), VX_MEM_WRITE, &output_buffer));
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

  for (uint32_t phase = 0; phase < phase_limit; ++phase) {
    for (uint32_t tile_id = 0; tile_id < kNumTiles; ++tile_id) {
      uint32_t phase_tile_id = phase * kNumTiles + tile_id;
      tma_descs[phase_tile_id] = {};
      tma_descs[phase_tile_id].addr = input_a_addr + phase_tile_id * kABytes;
      tma_descs[phase_tile_id].size_bytes = kABytes;
      tma_descs[phase_tile_id].tile_role = kTileRoleA;
      tma_descs[phase_tile_id].payload_kind = kHasSparse ? kPayloadSparsePayload : kPayloadDense;
      if constexpr (kHasSparse) {
        tma_descs[phase_tile_id].meta_addr = input_meta_addr + phase_tile_id * kMetaBytes;
        tma_descs[phase_tile_id].meta_size_bytes = kMetaBytes;
        tma_descs[phase_tile_id].meta_tmem_base = kMetaBankBase + (phase & 0x1);
        tma_descs[phase_tile_id].meta_col_span = kMetaBankSpan;
      }

      tma_descs[a_desc_count + phase_tile_id] = {};
      tma_descs[a_desc_count + phase_tile_id].addr = input_b_addr + phase_tile_id * kBBytes;
      tma_descs[a_desc_count + phase_tile_id].size_bytes = kBBytes;
      tma_descs[a_desc_count + phase_tile_id].tile_role = kTileRoleB;
      tma_descs[a_desc_count + phase_tile_id].payload_kind = kPayloadDense;
    }
  }
  for (uint32_t tile_id = 0; tile_id < kNumTiles; ++tile_id) {
    tma_descs[a_desc_count + b_desc_count + tile_id] = {};
    tma_descs[a_desc_count + b_desc_count + tile_id].addr = input_c_addr + tile_id * kCBytes;
    tma_descs[a_desc_count + b_desc_count + tile_id].size_bytes = kCBytes;
    tma_descs[a_desc_count + b_desc_count + tile_id].tile_role = kTileRoleC;
    tma_descs[a_desc_count + b_desc_count + tile_id].payload_kind = kPayloadDense;

    tma_descs[a_desc_count + b_desc_count + c_desc_count + tile_id] = {};
    tma_descs[a_desc_count + b_desc_count + c_desc_count + tile_id].addr = output_addr + tile_id * kCBytes;
    tma_descs[a_desc_count + b_desc_count + c_desc_count + tile_id].size_bytes = kCBytes;
    tma_descs[a_desc_count + b_desc_count + c_desc_count + tile_id].tile_role = kTileRoleC;
    tma_descs[a_desc_count + b_desc_count + c_desc_count + tile_id].payload_kind = kPayloadDense;
  }
  mma_descs[0].fmt_a = vt::ATYPE::id;
  mma_descs[0].fmt_b = vt::BTYPE::id;
  mma_descs[0].fmt_c = vt::OTYPE::id;
  mma_descs[0].sparse_mode = kSparse2To4 ? vt::sparse_2_4 : (kSparse1To4 ? vt::sparse_1_4 : vt::sparse_none);
  mma_descs[0].a_rows = kTileDim;
  mma_descs[0].a_cols = kKSlice;
  mma_descs[0].b_rows = kKSlice;
  mma_descs[0].b_cols = kTileDim;
  mma_descs[0].c_rows = kTileDim;
  mma_descs[0].c_cols = kTileDim;

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
  kernel_arg.phase_limit = phase_limit;
  kernel_arg.a_bank_span = kABankSpan;
  kernel_arg.b_bank_span = kBBankSpan;
  kernel_arg.c_bank_span = kCBankSpan;

  std::cout << "upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

  auto time_start = std::chrono::high_resolution_clock::now();

  RT_CHECK(vx_copy_to_dev(input_a_buffer, h_all_phase_a.data(), 0, h_all_phase_a.size()));
  if constexpr (kHasSparse) {
    RT_CHECK(vx_copy_to_dev(input_meta_buffer, h_all_phase_meta.data(), 0, h_all_phase_meta.size()));
  }
  RT_CHECK(vx_copy_to_dev(input_b_buffer, h_all_phase_b.data(), 0, h_all_phase_b.size()));
  RT_CHECK(vx_copy_to_dev(input_c_buffer, h_init_c.data(), 0, h_init_c.size()));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  RT_CHECK(vx_copy_from_dev(h_output_tiles.data(), output_buffer, 0, h_output_tiles.size()));

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  std::cout << "verify result (print-only)" << std::endl;
  std::vector<output_t> h_accum(kMatrixM * kMatrixN, host_utils::encode_output(0.0f));
  for (uint32_t tile_row = 0; tile_row < kTileRows; ++tile_row) {
    for (uint32_t tile_col = 0; tile_col < kTileCols; ++tile_col) {
      uint32_t tile_id = tile_row * kTileCols + tile_col;
      host_utils::scatter_c_tile_to_matrix(h_accum,
                                           h_output_tiles.data() + tile_id * kCBytes,
                                           kMatrixN,
                                           tile_row,
                                           tile_col);
    }
  }
  std::vector<float> h_result_float;
  host_utils::convert_output_matrix_to_float(h_result_float, h_accum);
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
