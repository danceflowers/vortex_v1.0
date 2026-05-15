#include "common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <rvfloats.h>
#include <util.h>
#include <vortex.h>

#include "tensor/open_tensorcore/tensor_compute/fp_types.h"

#define RT_CHECK(_expr)                                      \
  do {                                                       \
    int _ret = _expr;                                        \
    if (0 == _ret)                                           \
      break;                                                 \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
    cleanup();                                               \
    exit(-1);                                                \
  } while (false)

namespace {

struct tensor_map_host_t {
  uint64_t global_address;
  uint64_t box_size[5];
  uint64_t global_stride[5];
  uint32_t element_strides[5];
  uint8_t  element_type;
  uint8_t  interleave;
  uint8_t  swizzle;
  uint8_t  l2_promotion;
  uint8_t  oob_fill;
  uint8_t  rank;
  uint8_t  reserved0[2];
  uint32_t reserved1[3];
} __attribute__((packed));
static_assert(sizeof(tensor_map_host_t) == 128, "tensor_map_host_t must be 128B");

enum class Fmt : uint32_t {
  Fp8,
  Fp16,
  Fp32,
};

struct TestCase {
  uint32_t id;
  const char* name;
  Fmt a_fmt;
  Fmt b_fmt;
  Fmt c_fmt;
  Fmt d_fmt;
  uint32_t sparse_mode;
  double threshold;
};

struct ErrorMetrics {
  int errors = 0;
  double max_abs = 0.0;
  double mean_abs = 0.0;
  double rms = 0.0;
};

static const char* kernel_file = "kernel.vxbin";

vx_device_h device = nullptr;
vx_buffer_h krnl_buffer = nullptr;

uint32_t elem_bytes(Fmt fmt) {
  switch (fmt) {
  case Fmt::Fp8:  return 1;
  case Fmt::Fp16: return 2;
  case Fmt::Fp32: return 4;
  }
  return 0;
}

uint8_t tensor_map_element_type(Fmt fmt) {
  switch (fmt) {
  case Fmt::Fp8:  return 10;
  case Fmt::Fp16: return 6;
  case Fmt::Fp32: return 7;
  }
  return 0;
}

uint32_t encode_value(Fmt fmt, float value) {
  switch (fmt) {
  case Fmt::Fp8:
    return double_to_fp8_e4m3(value);
  case Fmt::Fp16:
    return rv_ftoh_s(vortex::bit_cast<uint32_t>(value), 0, nullptr);
  case Fmt::Fp32:
    return vortex::bit_cast<uint32_t>(value);
  }
  return 0;
}

float decode_value(Fmt fmt, uint32_t bits) {
  switch (fmt) {
  case Fmt::Fp8:
    return static_cast<float>(fp8_e4m3_to_double(static_cast<uint8_t>(bits & 0xff)));
  case Fmt::Fp16:
    return vortex::bit_cast<float>(
        rv_htof_s(static_cast<uint16_t>(bits & 0xffff), 0, nullptr));
  case Fmt::Fp32:
    return vortex::bit_cast<float>(bits);
  }
  return 0.0f;
}

void write_raw(std::vector<uint8_t>& bytes, uint32_t offset, Fmt fmt, uint32_t value) {
  uint32_t nbytes = elem_bytes(fmt);
  for (uint32_t b = 0; b < nbytes; ++b) {
    bytes.at(offset + b) = static_cast<uint8_t>((value >> (8 * b)) & 0xff);
  }
}

void pack_a_payload(Fmt fmt, const std::vector<uint32_t>& a, std::vector<uint8_t>* out) {
  out->assign(TCGEN05_PAYLOAD_BYTES, 0);
  uint32_t line_bytes = 64 * elem_bytes(fmt);
  for (uint32_t k_phase = 0; k_phase < 2; ++k_phase) {
    for (uint32_t storage_m = 0; storage_m < 2; ++storage_m) {
      uint32_t line = k_phase * 2 + storage_m;
      uint32_t base = line * line_bytes;
      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
          uint32_t row = storage_m * 8 + i;
          uint32_t col = k_phase * 8 + j;
          write_raw(*out, base + (i * 8 + j) * elem_bytes(fmt),
                    fmt, a.at(row * K_DIM + col));
        }
      }
    }
  }
}

void pack_b_payload(Fmt fmt, const std::vector<uint32_t>& b, std::vector<uint8_t>* out) {
  out->assign(TCGEN05_PAYLOAD_BYTES, 0);
  uint32_t line_bytes = 64 * elem_bytes(fmt);
  for (uint32_t k_phase = 0; k_phase < 2; ++k_phase) {
    for (uint32_t storage_n = 0; storage_n < 2; ++storage_n) {
      uint32_t line = k_phase * 2 + storage_n;
      uint32_t base = line * line_bytes;
      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
          uint32_t row = k_phase * 8 + i;
          uint32_t col = storage_n * 8 + j;
          write_raw(*out, base + (i * 8 + j) * elem_bytes(fmt),
                    fmt, b.at(row * N_DIM + col));
        }
      }
    }
  }
}

void pack_c_payload(Fmt fmt, const std::vector<uint32_t>& c, std::vector<uint8_t>* out) {
  uint32_t nbytes = elem_bytes(fmt);
  uint32_t rows_per_packet = 64 / (8 * nbytes);
  uint32_t packets_per_subtile = 8 / rows_per_packet;
  out->assign(4 * packets_per_subtile * 64, 0);
  for (uint32_t subtile = 0; subtile < 4; ++subtile) {
    uint32_t storage_m = subtile / 2;
    uint32_t storage_n = subtile % 2;
    for (uint32_t segment = 0; segment < packets_per_subtile; ++segment) {
      uint32_t packet_base = (subtile * packets_per_subtile + segment) * 64;
      for (uint32_t row_in_packet = 0; row_in_packet < rows_per_packet; ++row_in_packet) {
        uint32_t local_row = segment * rows_per_packet + row_in_packet;
        for (uint32_t col = 0; col < 8; ++col) {
          uint32_t global_row = storage_m * 8 + local_row;
          uint32_t global_col = storage_n * 8 + col;
          write_raw(*out, packet_base + (row_in_packet * 8 + col) * nbytes,
                    fmt, c.at(global_row * N_DIM + global_col));
        }
      }
    }
  }
}

void pack_sparse_a_payload_and_meta(uint32_t sparse_mode,
                                    const std::vector<uint32_t>& dense_a,
                                    std::vector<uint8_t>* a_payload,
                                    std::vector<uint8_t>* meta_payload) {
  std::vector<uint32_t> compact_a(M_DIM * K_DIM, 0);
  meta_payload->assign(TCGEN05_META_BYTES, 0);

  for (uint32_t storage_m = 0; storage_m < 2; ++storage_m) {
    for (uint32_t k_phase = 0; k_phase < 2; ++k_phase) {
      uint32_t line_index = storage_m * 2 + k_phase;
      for (uint32_t row = 0; row < 8; ++row) {
        uint32_t global_row = storage_m * 8 + row;
        uint16_t row_meta = 0;
        uint32_t payload_cursor = 0;
        for (uint32_t group = 0; group < 2; ++group) {
          uint32_t dense_col_base = k_phase * 8 + group * 4;
          uint32_t picks[2] = {0, 0};
          uint32_t values[2] = {0, 0};
          uint32_t pick_count = 0;
          for (uint32_t lane = 0; lane < 4; ++lane) {
            uint32_t value = dense_a.at(global_row * K_DIM + dense_col_base + lane);
            if (value == 0) {
              continue;
            }
            if (pick_count >= 2) {
              std::abort();
            }
            picks[pick_count] = lane;
            values[pick_count] = value;
            ++pick_count;
          }

          uint32_t expected = (sparse_mode == 1) ? 2 : 1;
          if (pick_count != expected) {
            std::abort();
          }
          for (uint32_t p = 0; p < expected; ++p) {
            compact_a.at(global_row * K_DIM + k_phase * 8 + payload_cursor++) = values[p];
          }

          if (sparse_mode == 1) {
            row_meta |= static_cast<uint16_t>(picks[0] & 0x3u) << (group * 4 + 0);
            row_meta |= static_cast<uint16_t>(picks[1] & 0x3u) << (group * 4 + 2);
          } else {
            row_meta |= static_cast<uint16_t>(picks[0] & 0x3u) << (group * 2);
          }
        }
        meta_payload->at(line_index * 16 + row * 2 + 0) =
            static_cast<uint8_t>(row_meta & 0xffu);
        meta_payload->at(line_index * 16 + row * 2 + 1) =
            static_cast<uint8_t>((row_meta >> 8) & 0xffu);
      }
    }
  }

  pack_a_payload(Fmt::Fp8, compact_a, a_payload);
}

uint32_t tensor_map_element_bytes(uint8_t element_type) {
  switch (element_type) {
  case 6:  return 2;
  case 7:  return 4;
  case 10: return 1;
  default: return 1;
  }
}

tensor_map_host_t make_tmap(uint64_t payload_addr, uint32_t bytes, uint8_t element_type) {
  uint32_t nbytes = tensor_map_element_bytes(element_type);
  tensor_map_host_t tmap{};
  tmap.global_address = payload_addr;
  tmap.box_size[0] = bytes / nbytes;
  tmap.global_stride[0] = nbytes;
  tmap.element_strides[0] = 1;
  tmap.element_type = element_type;
  tmap.rank = 1;
  return tmap;
}

void build_asym_case(std::vector<uint32_t>* a_bits,
                     std::vector<uint32_t>* b_bits,
                     std::vector<uint32_t>* c_bits,
                     std::vector<float>* a_fp32,
                     std::vector<float>* b_fp32,
                     std::vector<float>* c_fp32,
                     std::vector<uint8_t>* a_payload,
                     std::vector<uint8_t>* b_payload,
                     std::vector<uint8_t>* c_payload,
                     std::vector<uint8_t>* meta_payload) {
  a_bits->assign(M_DIM * K_DIM, 0);
  b_bits->assign(K_DIM * N_DIM, 0);
  c_bits->assign(M_DIM * N_DIM, 0);
  a_fp32->assign(M_DIM * K_DIM, 0.0f);
  b_fp32->assign(K_DIM * N_DIM, 0.0f);
  c_fp32->assign(M_DIM * N_DIM, 0.0f);
  meta_payload->assign(TCGEN05_META_BYTES, 0);
  for (uint32_t m = 0; m < M_DIM; ++m) {
    for (uint32_t k = 0; k < K_DIM; ++k) {
      float value = (m == k) ? 1.0f : 0.0f;
      a_fp32->at(m * K_DIM + k) = value;
      a_bits->at(m * K_DIM + k) = encode_value(Fmt::Fp16, value);
    }
  }
  for (uint32_t k = 0; k < K_DIM; ++k) {
    for (uint32_t n = 0; n < N_DIM; ++n) {
      float value = static_cast<float>(((k * 5 + n * 3) & 0x7) + 1) / 16.0f;
      b_fp32->at(k * N_DIM + n) = value;
      b_bits->at(k * N_DIM + n) = encode_value(Fmt::Fp8, value);
    }
  }
  for (uint32_t m = 0; m < M_DIM; ++m) {
    for (uint32_t n = 0; n < N_DIM; ++n) {
      float value = static_cast<float>(((m * 7 + n * 3) & 0x7) + 1) / 64.0f;
      c_fp32->at(m * N_DIM + n) = value;
      c_bits->at(m * N_DIM + n) = encode_value(Fmt::Fp32, value);
    }
  }
  pack_a_payload(Fmt::Fp16, *a_bits, a_payload);
  pack_b_payload(Fmt::Fp8, *b_bits, b_payload);
  pack_c_payload(Fmt::Fp32, *c_bits, c_payload);
}

void build_sparse_case(uint32_t sparse_mode,
                       std::vector<uint32_t>* a_bits,
                       std::vector<uint32_t>* b_bits,
                       std::vector<uint32_t>* c_bits,
                       std::vector<float>* a_fp32,
                       std::vector<float>* b_fp32,
                       std::vector<float>* c_fp32,
                       std::vector<uint8_t>* a_payload,
                       std::vector<uint8_t>* b_payload,
                       std::vector<uint8_t>* c_payload,
                       std::vector<uint8_t>* meta_payload) {
  a_bits->assign(M_DIM * K_DIM, 0);
  b_bits->assign(K_DIM * N_DIM, 0);
  c_bits->assign(M_DIM * N_DIM, 0);
  a_fp32->assign(M_DIM * K_DIM, 0.0f);
  b_fp32->assign(K_DIM * N_DIM, 0.0f);
  c_fp32->assign(M_DIM * N_DIM, 0.0f);
  for (uint32_t m = 0; m < M_DIM; ++m) {
    for (uint32_t group = 0; group < K_DIM / 4; ++group) {
      uint32_t base = group * 4;
      if (sparse_mode == 1) {
        uint32_t p0 = (m + group) & 0x3;
        uint32_t p1 = (p0 + 2) & 0x3;
        float v0 = static_cast<float>(((m + group) & 0x3) + 1) / 16.0f;
        float v1 = static_cast<float>(((m + 2 * group) & 0x3) + 1) / 32.0f;
        a_fp32->at(m * K_DIM + base + p0) = v0;
        a_fp32->at(m * K_DIM + base + p1) = v1;
        a_bits->at(m * K_DIM + base + p0) = encode_value(Fmt::Fp8, v0);
        a_bits->at(m * K_DIM + base + p1) = encode_value(Fmt::Fp8, v1);
      } else {
        uint32_t p = (m + group) & 0x3;
        float v = static_cast<float>(((m + group) & 0x3) + 1) / 16.0f;
        a_fp32->at(m * K_DIM + base + p) = v;
        a_bits->at(m * K_DIM + base + p) = encode_value(Fmt::Fp8, v);
      }
    }
  }
  for (uint32_t k = 0; k < K_DIM; ++k) {
    for (uint32_t n = 0; n < N_DIM; ++n) {
      float value = static_cast<float>(((k * 3 + n * 5) & 0x7) + 1) / 32.0f;
      b_fp32->at(k * N_DIM + n) = value;
      b_bits->at(k * N_DIM + n) = encode_value(Fmt::Fp8, value);
    }
  }
  for (uint32_t m = 0; m < M_DIM; ++m) {
    for (uint32_t n = 0; n < N_DIM; ++n) {
      float value = static_cast<float>(((m * 5 + n * 3) & 0x7) + 1) / 64.0f;
      c_fp32->at(m * N_DIM + n) = value;
      c_bits->at(m * N_DIM + n) = encode_value(Fmt::Fp16, value);
    }
  }
  pack_sparse_a_payload_and_meta(sparse_mode, *a_bits, a_payload, meta_payload);
  pack_b_payload(Fmt::Fp8, *b_bits, b_payload);
  pack_c_payload(Fmt::Fp16, *c_bits, c_payload);
}

std::vector<float> build_quantized_fp32_reference(const TestCase& tc,
                                                  const std::vector<uint32_t>& a_bits,
                                                  const std::vector<uint32_t>& b_bits,
                                                  const std::vector<uint32_t>& c_bits) {
  std::vector<float> ref(M_DIM * N_DIM, 0.0f);
  for (uint32_t m = 0; m < M_DIM; ++m) {
    for (uint32_t n = 0; n < N_DIM; ++n) {
      float acc = decode_value(tc.c_fmt, c_bits.at(m * N_DIM + n));
      for (uint32_t k = 0; k < K_DIM; ++k) {
        float a = decode_value(tc.a_fmt, a_bits.at(m * K_DIM + k));
        float b = decode_value(tc.b_fmt, b_bits.at(k * N_DIM + n));
        acc += a * b;
      }
      ref.at(m * N_DIM + n) = acc;
    }
  }
  return ref;
}

std::vector<float> build_raw_fp32_reference(const std::vector<float>& a_fp32,
                                            const std::vector<float>& b_fp32,
                                            const std::vector<float>& c_fp32) {
  std::vector<float> ref(M_DIM * N_DIM, 0.0f);
  for (uint32_t m = 0; m < M_DIM; ++m) {
    for (uint32_t n = 0; n < N_DIM; ++n) {
      float acc = c_fp32.at(m * N_DIM + n);
      for (uint32_t k = 0; k < K_DIM; ++k) {
        acc += a_fp32.at(m * K_DIM + k) * b_fp32.at(k * N_DIM + n);
      }
      ref.at(m * N_DIM + n) = acc;
    }
  }
  return ref;
}

std::vector<float> decode_output_words(const std::vector<uint32_t>& words, Fmt fmt) {
  std::vector<float> out(M_DIM * N_DIM, 0.0f);
  uint32_t nbytes = elem_bytes(fmt);
  for (uint32_t elem = 0; elem < M_DIM * N_DIM; ++elem) {
    uint32_t lane = elem % 32;
    uint32_t chunk = elem / 32;
    uint32_t lane_byte_offset = chunk * nbytes;
    uint32_t word_idx = lane_byte_offset / sizeof(uint32_t);
    uint32_t word_off = lane_byte_offset % sizeof(uint32_t);
    uint32_t word = words.at(word_idx * 32 + lane);
    uint32_t bits = (word >> (8 * word_off)) & ((1ull << (8 * nbytes)) - 1ull);
    out.at(elem) = decode_value(fmt, bits);
  }
  return out;
}

ErrorMetrics compare_reference(const std::vector<float>& got,
                               const std::vector<float>& ref,
                               double threshold,
                               const char* label) {
  ErrorMetrics metrics{};
  double sum_abs = 0.0;
  double sum_sq = 0.0;
  for (uint32_t i = 0; i < got.size(); ++i) {
    double diff = std::fabs(static_cast<double>(got.at(i)) - static_cast<double>(ref.at(i)));
    sum_abs += diff;
    sum_sq += diff * diff;
    metrics.max_abs = std::max(metrics.max_abs, diff);
    if (diff > threshold) {
      if (metrics.errors < 8) {
        std::cout << "  " << label << "_mismatch[" << i << "]: got=" << got.at(i)
                  << " ref=" << ref.at(i) << " diff=" << diff << std::endl;
      }
      ++metrics.errors;
    }
  }

  metrics.mean_abs = sum_abs / static_cast<double>(got.size());
  metrics.rms = std::sqrt(sum_sq / static_cast<double>(got.size()));
  std::cout << "  " << label << "_ERROR_METRICS: max_abs_error=" << metrics.max_abs
            << ", mean_abs_error=" << metrics.mean_abs
            << ", rms_error=" << metrics.rms
            << ", threshold=" << threshold << std::endl;
  return metrics;
}

} // namespace

void cleanup() {
  if (device) {
    if (krnl_buffer) {
      vx_mem_free(krnl_buffer);
      krnl_buffer = nullptr;
    }
    vx_dev_close(device);
    device = nullptr;
  }
}

int run_case(const TestCase& tc) {
  std::vector<uint32_t> a_bits;
  std::vector<uint32_t> b_bits;
  std::vector<uint32_t> c_bits;
  std::vector<float> a_fp32;
  std::vector<float> b_fp32;
  std::vector<float> c_fp32;
  std::vector<uint8_t> a_payload;
  std::vector<uint8_t> b_payload;
  std::vector<uint8_t> c_payload;
  std::vector<uint8_t> meta_payload;

  if (tc.id == TCGEN05_CASE_ASYM_F16_F8_F32) {
    build_asym_case(&a_bits, &b_bits, &c_bits, &a_fp32, &b_fp32, &c_fp32,
                    &a_payload, &b_payload, &c_payload, &meta_payload);
  } else {
    build_sparse_case(tc.sparse_mode, &a_bits, &b_bits, &c_bits, &a_fp32, &b_fp32, &c_fp32,
                      &a_payload, &b_payload, &c_payload, &meta_payload);
  }
  auto golden1 = build_quantized_fp32_reference(tc, a_bits, b_bits, c_bits);
  auto golden2 = build_raw_fp32_reference(a_fp32, b_fp32, c_fp32);

  vx_buffer_h a_payload_buffer = nullptr;
  vx_buffer_h b_payload_buffer = nullptr;
  vx_buffer_h c_payload_buffer = nullptr;
  vx_buffer_h meta_payload_buffer = nullptr;
  vx_buffer_h a_tmap_buffer = nullptr;
  vx_buffer_h b_tmap_buffer = nullptr;
  vx_buffer_h c_tmap_buffer = nullptr;
  vx_buffer_h meta_tmap_buffer = nullptr;
  vx_buffer_h out_buffer = nullptr;
  vx_buffer_h args_buffer = nullptr;
  auto free_case_buffers = [&]() {
    if (a_payload_buffer) vx_mem_free(a_payload_buffer);
    if (b_payload_buffer) vx_mem_free(b_payload_buffer);
    if (c_payload_buffer) vx_mem_free(c_payload_buffer);
    if (meta_payload_buffer) vx_mem_free(meta_payload_buffer);
    if (a_tmap_buffer) vx_mem_free(a_tmap_buffer);
    if (b_tmap_buffer) vx_mem_free(b_tmap_buffer);
    if (c_tmap_buffer) vx_mem_free(c_tmap_buffer);
    if (meta_tmap_buffer) vx_mem_free(meta_tmap_buffer);
    if (out_buffer) vx_mem_free(out_buffer);
    if (args_buffer) vx_mem_free(args_buffer);
  };

  RT_CHECK(vx_mem_alloc(device, a_payload.size(), VX_MEM_READ, &a_payload_buffer));
  RT_CHECK(vx_mem_alloc(device, b_payload.size(), VX_MEM_READ, &b_payload_buffer));
  RT_CHECK(vx_mem_alloc(device, c_payload.size(), VX_MEM_READ, &c_payload_buffer));
  RT_CHECK(vx_mem_alloc(device, meta_payload.size(), VX_MEM_READ, &meta_payload_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(tensor_map_host_t), VX_MEM_READ, &a_tmap_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(tensor_map_host_t), VX_MEM_READ, &b_tmap_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(tensor_map_host_t), VX_MEM_READ, &c_tmap_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(tensor_map_host_t), VX_MEM_READ, &meta_tmap_buffer));

  std::vector<uint32_t> out_init(NUM_TCU_LD_WORDS * 32, 0xdeadbeefu);
  RT_CHECK(vx_mem_alloc(device, out_init.size() * sizeof(uint32_t), VX_MEM_WRITE, &out_buffer));

  uint64_t a_payload_addr = 0;
  uint64_t b_payload_addr = 0;
  uint64_t c_payload_addr = 0;
  uint64_t meta_payload_addr = 0;
  uint64_t a_tmap_addr = 0;
  uint64_t b_tmap_addr = 0;
  uint64_t c_tmap_addr = 0;
  uint64_t meta_tmap_addr = 0;
  uint64_t out_addr = 0;
  RT_CHECK(vx_mem_address(a_payload_buffer, &a_payload_addr));
  RT_CHECK(vx_mem_address(b_payload_buffer, &b_payload_addr));
  RT_CHECK(vx_mem_address(c_payload_buffer, &c_payload_addr));
  RT_CHECK(vx_mem_address(meta_payload_buffer, &meta_payload_addr));
  RT_CHECK(vx_mem_address(a_tmap_buffer, &a_tmap_addr));
  RT_CHECK(vx_mem_address(b_tmap_buffer, &b_tmap_addr));
  RT_CHECK(vx_mem_address(c_tmap_buffer, &c_tmap_addr));
  RT_CHECK(vx_mem_address(meta_tmap_buffer, &meta_tmap_addr));
  RT_CHECK(vx_mem_address(out_buffer, &out_addr));

  auto a_tmap = make_tmap(a_payload_addr, a_payload.size(), tensor_map_element_type(tc.a_fmt));
  auto b_tmap = make_tmap(b_payload_addr, b_payload.size(), tensor_map_element_type(tc.b_fmt));
  auto c_tmap = make_tmap(c_payload_addr, c_payload.size(), tensor_map_element_type(tc.c_fmt));
  auto meta_tmap = make_tmap(meta_payload_addr, meta_payload.size(), 0);
  RT_CHECK(vx_copy_to_dev(a_payload_buffer, a_payload.data(), 0, a_payload.size()));
  RT_CHECK(vx_copy_to_dev(b_payload_buffer, b_payload.data(), 0, b_payload.size()));
  RT_CHECK(vx_copy_to_dev(c_payload_buffer, c_payload.data(), 0, c_payload.size()));
  RT_CHECK(vx_copy_to_dev(meta_payload_buffer, meta_payload.data(), 0, meta_payload.size()));
  RT_CHECK(vx_copy_to_dev(a_tmap_buffer, &a_tmap, 0, sizeof(a_tmap)));
  RT_CHECK(vx_copy_to_dev(b_tmap_buffer, &b_tmap, 0, sizeof(b_tmap)));
  RT_CHECK(vx_copy_to_dev(c_tmap_buffer, &c_tmap, 0, sizeof(c_tmap)));
  RT_CHECK(vx_copy_to_dev(meta_tmap_buffer, &meta_tmap, 0, sizeof(meta_tmap)));
  RT_CHECK(vx_copy_to_dev(out_buffer, out_init.data(), 0, out_init.size() * sizeof(uint32_t)));

  kernel_arg_t kernel_arg{};
  kernel_arg.grid_dim[0] = 1;
  kernel_arg.grid_dim[1] = 1;
  kernel_arg.grid_dim[2] = 1;
  kernel_arg.block_dim[0] = NUM_THREADS;
  kernel_arg.block_dim[1] = 1;
  kernel_arg.block_dim[2] = 1;
  kernel_arg.a_tmap_addr = a_tmap_addr;
  kernel_arg.b_tmap_addr = b_tmap_addr;
  kernel_arg.c_tmap_addr = c_tmap_addr;
  kernel_arg.meta_tmap_addr = meta_tmap_addr;
  kernel_arg.out_addr = out_addr;
  kernel_arg.case_id = tc.id;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

  std::cout << "[Vortex][case " << tc.id << "] " << tc.name << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  std::vector<uint32_t> got_words(NUM_TCU_LD_WORDS * 32);
  RT_CHECK(vx_copy_from_dev(got_words.data(), out_buffer, 0,
                            got_words.size() * sizeof(uint32_t)));
  auto got = decode_output_words(got_words, tc.d_fmt);

  auto golden1_metrics = compare_reference(got, golden1, tc.threshold,
                                           "GOLDEN1_QUANTIZED_INPUT");
  auto golden2_metrics = compare_reference(got, golden2, tc.threshold,
                                           "GOLDEN2_RAW_FP32_INPUT");

  free_case_buffers();
  if (golden1_metrics.errors == 0 && golden2_metrics.errors == 0) {
    std::cout << "  PASS" << std::endl;
    return 0;
  }
  std::cout << "  FAILED: "
            << golden1_metrics.errors << " golden1 mismatches, "
            << golden2_metrics.errors << " golden2 mismatches" << std::endl;
  return 1;
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

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  const TestCase tests[] = {
    {TCGEN05_CASE_ASYM_F16_F8_F32,
     "tcgen05_dense_fp16_fp8_to_fp32",
     Fmt::Fp16, Fmt::Fp8, Fmt::Fp32, Fmt::Fp32, 0, 0.01},
    {TCGEN05_CASE_SPARSE_2_4_F8,
     "tcgen05_sparse_2_4_fp8_fp8_to_fp16",
     Fmt::Fp8, Fmt::Fp8, Fmt::Fp16, Fmt::Fp16, 1, 0.05},
    {TCGEN05_CASE_SPARSE_1_4_F8,
     "tcgen05_sparse_1_4_fp8_fp8_to_fp16",
     Fmt::Fp8, Fmt::Fp8, Fmt::Fp16, Fmt::Fp16, 2, 0.05},
  };

  int failures = 0;
  for (const auto& tc : tests) {
    failures += run_case(tc);
  }
  cleanup();

  if (failures == 0) {
    std::cout << "PASSED! extended tcgen05 MMA cases matched golden1 and golden2" << std::endl;
    return 0;
  }
  std::cout << "FAILED: " << failures << " cases failed" << std::endl;
  return 1;
}
