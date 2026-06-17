#include "common.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <rvfloats.h>
#include <util.h>
#include <vortex.h>

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

static const char* kernel_file = "kernel.vxbin";

vx_device_h device = nullptr;
vx_buffer_h a_payload_buffer = nullptr;
vx_buffer_h a_poison_payload_buffer = nullptr;
vx_buffer_h b_payload_buffer = nullptr;
vx_buffer_h a_tmap_buffer = nullptr;
vx_buffer_h a_poison_tmap_buffer = nullptr;
vx_buffer_h b_tmap_buffer = nullptr;
vx_buffer_h out_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;

static uint16_t f32_to_fp16(float v) {
  return rv_ftoh_s(vortex::bit_cast<uint32_t>(v), 0, nullptr);
}

static float word_to_float(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static void write_fp16(std::vector<uint8_t>& bytes, uint32_t offset, uint16_t value) {
  bytes.at(offset + 0) = static_cast<uint8_t>(value & 0xff);
  bytes.at(offset + 1) = static_cast<uint8_t>((value >> 8) & 0xff);
}

static void pack_a_payload(const uint16_t a[M_DIM][K_DIM], std::vector<uint8_t>* out) {
  out->assign(TCGEN05_PAYLOAD_BYTES, 0);
  for (uint32_t k_phase = 0; k_phase < 2; ++k_phase) {
    for (uint32_t storage_m = 0; storage_m < 2; ++storage_m) {
      uint32_t line = k_phase * 2 + storage_m;
      uint32_t base = line * 128;
      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
          uint32_t row = storage_m * 8 + i;
          uint32_t col = k_phase * 8 + j;
          write_fp16(*out, base + (i * 8 + j) * 2, a[row][col]);
        }
      }
    }
  }
}

static void pack_b_payload(const uint16_t b[K_DIM][N_DIM], std::vector<uint8_t>* out) {
  out->assign(TCGEN05_PAYLOAD_BYTES, 0);
  for (uint32_t k_phase = 0; k_phase < 2; ++k_phase) {
    for (uint32_t storage_n = 0; storage_n < 2; ++storage_n) {
      uint32_t line = k_phase * 2 + storage_n;
      uint32_t base = line * 128;
      for (uint32_t i = 0; i < 8; ++i) {
        for (uint32_t j = 0; j < 8; ++j) {
          uint32_t row = k_phase * 8 + i;
          uint32_t col = storage_n * 8 + j;
          write_fp16(*out, base + (i * 8 + j) * 2, b[row][col]);
        }
      }
    }
  }
}

static tensor_map_host_t make_payload_tmap(uint64_t payload_addr) {
  tensor_map_host_t tmap{};
  tmap.global_address = payload_addr;
  tmap.box_size[0] = TCGEN05_PAYLOAD_BYTES / sizeof(uint16_t);
  tmap.global_stride[0] = sizeof(uint16_t);
  tmap.element_strides[0] = 1;
  tmap.element_type = 6;
  tmap.rank = 1;
  return tmap;
}

static void free_buffer(vx_buffer_h* buffer) {
  if (*buffer) {
    vx_mem_free(*buffer);
    *buffer = nullptr;
  }
}

} // namespace

void cleanup() {
  if (device) {
    free_buffer(&a_payload_buffer);
    free_buffer(&a_poison_payload_buffer);
    free_buffer(&b_payload_buffer);
    free_buffer(&a_tmap_buffer);
    free_buffer(&a_poison_tmap_buffer);
    free_buffer(&b_tmap_buffer);
    free_buffer(&out_buffer);
    free_buffer(&krnl_buffer);
    free_buffer(&args_buffer);
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

  float a_fp32[M_DIM][K_DIM] = {};
  float b_fp32[K_DIM][N_DIM] = {};
  uint16_t a[M_DIM][K_DIM] = {};
  uint16_t a_poison[M_DIM][K_DIM] = {};
  uint16_t b[K_DIM][N_DIM] = {};
  std::vector<float> ref(M_DIM * N_DIM, 0.0f);

  for (uint32_t m = 0; m < M_DIM; ++m) {
    for (uint32_t k = 0; k < K_DIM; ++k) {
      a_fp32[m][k] = (m == k) ? 1.0f : 0.0f;
      a[m][k] = f32_to_fp16(a_fp32[m][k]);
      a_poison[m][k] = f32_to_fp16(0.0f);
    }
  }

  for (uint32_t k = 0; k < K_DIM; ++k) {
    for (uint32_t n = 0; n < N_DIM; ++n) {
      b_fp32[k][n] = static_cast<float>(((k * 7 + n * 3) & 0xf) + 1) / 32.0f;
      b[k][n] = f32_to_fp16(b_fp32[k][n]);
    }
  }

  for (uint32_t m = 0; m < M_DIM; ++m) {
    for (uint32_t n = 0; n < N_DIM; ++n) {
      float acc = 0.0f;
      for (uint32_t k = 0; k < K_DIM; ++k) {
        acc += a_fp32[m][k] * b_fp32[k][n];
      }
      ref.at(m * N_DIM + n) = acc;
    }
  }

  std::vector<uint8_t> a_payload;
  std::vector<uint8_t> a_poison_payload;
  std::vector<uint8_t> b_payload;
  pack_a_payload(a, &a_payload);
  pack_a_payload(a_poison, &a_poison_payload);
  pack_b_payload(b, &b_payload);

  uint32_t out_words = COLLECTOR_RESULT_COUNT * NUM_TCU_LD_WORDS * 32;
  std::vector<uint32_t> out_init(out_words, 0xdeadbeefu);
  RT_CHECK(vx_mem_alloc(device, a_payload.size(), VX_MEM_READ, &a_payload_buffer));
  RT_CHECK(vx_mem_alloc(device, a_poison_payload.size(), VX_MEM_READ, &a_poison_payload_buffer));
  RT_CHECK(vx_mem_alloc(device, b_payload.size(), VX_MEM_READ, &b_payload_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(tensor_map_host_t), VX_MEM_READ, &a_tmap_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(tensor_map_host_t), VX_MEM_READ, &a_poison_tmap_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(tensor_map_host_t), VX_MEM_READ, &b_tmap_buffer));
  RT_CHECK(vx_mem_alloc(device, out_init.size() * sizeof(uint32_t), VX_MEM_WRITE, &out_buffer));

  uint64_t a_payload_addr = 0;
  uint64_t a_poison_payload_addr = 0;
  uint64_t b_payload_addr = 0;
  uint64_t a_tmap_addr = 0;
  uint64_t a_poison_tmap_addr = 0;
  uint64_t b_tmap_addr = 0;
  uint64_t out_addr = 0;
  RT_CHECK(vx_mem_address(a_payload_buffer, &a_payload_addr));
  RT_CHECK(vx_mem_address(a_poison_payload_buffer, &a_poison_payload_addr));
  RT_CHECK(vx_mem_address(b_payload_buffer, &b_payload_addr));
  RT_CHECK(vx_mem_address(a_tmap_buffer, &a_tmap_addr));
  RT_CHECK(vx_mem_address(a_poison_tmap_buffer, &a_poison_tmap_addr));
  RT_CHECK(vx_mem_address(b_tmap_buffer, &b_tmap_addr));
  RT_CHECK(vx_mem_address(out_buffer, &out_addr));

  auto a_tmap = make_payload_tmap(a_payload_addr);
  auto a_poison_tmap = make_payload_tmap(a_poison_payload_addr);
  auto b_tmap = make_payload_tmap(b_payload_addr);
  RT_CHECK(vx_copy_to_dev(a_payload_buffer, a_payload.data(), 0, a_payload.size()));
  RT_CHECK(vx_copy_to_dev(a_poison_payload_buffer,
                          a_poison_payload.data(),
                          0,
                          a_poison_payload.size()));
  RT_CHECK(vx_copy_to_dev(b_payload_buffer, b_payload.data(), 0, b_payload.size()));
  RT_CHECK(vx_copy_to_dev(a_tmap_buffer, &a_tmap, 0, sizeof(a_tmap)));
  RT_CHECK(vx_copy_to_dev(a_poison_tmap_buffer, &a_poison_tmap, 0, sizeof(a_poison_tmap)));
  RT_CHECK(vx_copy_to_dev(b_tmap_buffer, &b_tmap, 0, sizeof(b_tmap)));
  RT_CHECK(vx_copy_to_dev(out_buffer, out_init.data(), 0, out_init.size() * sizeof(uint32_t)));

  kernel_arg_t kernel_arg{};
  kernel_arg.grid_dim[0] = 1;
  kernel_arg.grid_dim[1] = 1;
  kernel_arg.grid_dim[2] = 1;
  kernel_arg.block_dim[0] = NUM_THREADS;
  kernel_arg.block_dim[1] = 1;
  kernel_arg.block_dim[2] = 1;
  kernel_arg.a_tmap_addr = a_tmap_addr;
  kernel_arg.a_poison_tmap_addr = a_poison_tmap_addr;
  kernel_arg.b_tmap_addr = b_tmap_addr;
  kernel_arg.out_addr = out_addr;

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  std::vector<uint32_t> got_words(out_words);
  RT_CHECK(vx_copy_from_dev(got_words.data(),
                            out_buffer,
                            0,
                            got_words.size() * sizeof(uint32_t)));
  cleanup();

  const char* labels[COLLECTOR_RESULT_COUNT] = {"FILL_USE_LASTUSE"};
  int errors = 0;
  double max_abs = 0.0;
  double sum_abs = 0.0;
  double sum_sq = 0.0;
  uint32_t samples = 0;
  for (uint32_t result = 0; result < COLLECTOR_RESULT_COUNT; ++result) {
    for (uint32_t elem = 0; elem < M_DIM * N_DIM; ++elem) {
      uint32_t got_idx = result * NUM_TCU_LD_WORDS * 32 + elem;
      float got = word_to_float(got_words.at(got_idx));
      float expected = ref.at(elem);
      double diff = std::fabs(static_cast<double>(got) - static_cast<double>(expected));
      max_abs = std::max(max_abs, diff);
      sum_abs += diff;
      sum_sq += diff * diff;
      ++samples;
      if (diff > 0.01) {
        if (errors < 12) {
          std::cout << labels[result] << "_mismatch[" << elem << "]: got=" << got
                    << " ref=" << expected << " diff=" << diff << std::endl;
        }
        ++errors;
      }
    }
  }

  double mean_abs = sum_abs / static_cast<double>(samples);
  double rms = std::sqrt(sum_sq / static_cast<double>(samples));
  std::cout << "COLLECTOR_ABUF_ERROR_METRICS: max_abs_error=" << max_abs
            << ", mean_abs_error=" << mean_abs
            << ", rms_error=" << rms
            << ", threshold=0.01" << std::endl;

  if (errors == 0) {
    std::cout << "PASSED! tcgen05 collector buffer FILL/USE/LASTUSE reused ABuf"
              << std::endl;
    return 0;
  }

  std::cout << "FAILED: " << errors << " collector buffer mismatches" << std::endl;
  return 1;
}
