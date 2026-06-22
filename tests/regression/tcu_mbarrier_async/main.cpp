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
vx_buffer_h b_payload_buffer = nullptr;
vx_buffer_h a_tmap_buffer = nullptr;
vx_buffer_h b_tmap_buffer = nullptr;
vx_buffer_h out_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static inline uint16_t f32_to_fp16(float v) {
  return rv_ftoh_s(vortex::bit_cast<uint32_t>(v), 0, nullptr);
}

static inline float fp16_to_f32(uint16_t b) {
  return vortex::bit_cast<float>(rv_htof_s(b, 0, nullptr));
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
  tmap.box_size[0] = (TCGEN05_PAYLOAD_BYTES / sizeof(uint16_t));
  tmap.global_stride[0] = sizeof(uint16_t);
  tmap.element_strides[0] = 1;
  tmap.element_type = 6; // F16
  tmap.rank = 1;
  return tmap;
}

} // namespace

void cleanup() {
  if (device) {
    if (a_payload_buffer) vx_mem_free(a_payload_buffer);
    if (b_payload_buffer) vx_mem_free(b_payload_buffer);
    if (a_tmap_buffer)    vx_mem_free(a_tmap_buffer);
    if (b_tmap_buffer)    vx_mem_free(b_tmap_buffer);
    if (out_buffer)       vx_mem_free(out_buffer);
    if (krnl_buffer)      vx_mem_free(krnl_buffer);
    if (args_buffer)      vx_mem_free(args_buffer);
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
  uint16_t b[K_DIM][N_DIM] = {};
  std::vector<float> ref(M_DIM * N_DIM, 0.0f);
  for (uint32_t m = 0; m < M_DIM; ++m) {
    for (uint32_t k = 0; k < K_DIM; ++k) {
      // Cyclic-shift permutation A so D[m][n] = B[(m+1)%K][n] (a non-trivial
      // matmul, single non-zero per row → exact, no accumulation error).
      a_fp32[m][k] = (k == (m + 1) % K_DIM) ? 1.0f : 0.0f;
      a[m][k] = f32_to_fp16(a_fp32[m][k]);
    }
  }
  for (uint32_t k = 0; k < K_DIM; ++k) {
    for (uint32_t n = 0; n < N_DIM; ++n) {
      b_fp32[k][n] = static_cast<float>(((k * 3 + n) & 0x7) + 1) / 100.0f;
      b[k][n] = f32_to_fp16(b_fp32[k][n]);
    }
  }
  for (uint32_t m = 0; m < M_DIM; ++m) {
    for (uint32_t n = 0; n < N_DIM; ++n) {
      float acc = 0.0f;
      for (uint32_t k = 0; k < K_DIM; ++k) {
        acc += a_fp32[m][k] * b_fp32[k][n];
      }
      ref[m * N_DIM + n] = acc;
    }
  }

  std::vector<uint8_t> a_payload;
  std::vector<uint8_t> b_payload;
  pack_a_payload(a, &a_payload);
  pack_b_payload(b, &b_payload);
  std::vector<float> out_init(M_DIM * N_DIM, -1.0f);

  RT_CHECK(vx_mem_alloc(device, a_payload.size(), VX_MEM_READ, &a_payload_buffer));
  RT_CHECK(vx_mem_alloc(device, b_payload.size(), VX_MEM_READ, &b_payload_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(tensor_map_host_t), VX_MEM_READ, &a_tmap_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(tensor_map_host_t), VX_MEM_READ, &b_tmap_buffer));
  RT_CHECK(vx_mem_alloc(device, out_init.size() * sizeof(float), VX_MEM_WRITE, &out_buffer));

  uint64_t a_payload_addr = 0, b_payload_addr = 0, a_tmap_addr = 0, b_tmap_addr = 0, out_addr = 0;
  RT_CHECK(vx_mem_address(a_payload_buffer, &a_payload_addr));
  RT_CHECK(vx_mem_address(b_payload_buffer, &b_payload_addr));
  RT_CHECK(vx_mem_address(a_tmap_buffer, &a_tmap_addr));
  RT_CHECK(vx_mem_address(b_tmap_buffer, &b_tmap_addr));
  RT_CHECK(vx_mem_address(out_buffer, &out_addr));

  auto a_tmap = make_payload_tmap(a_payload_addr);
  auto b_tmap = make_payload_tmap(b_payload_addr);
  RT_CHECK(vx_copy_to_dev(a_payload_buffer, a_payload.data(), 0, a_payload.size()));
  RT_CHECK(vx_copy_to_dev(b_payload_buffer, b_payload.data(), 0, b_payload.size()));
  RT_CHECK(vx_copy_to_dev(a_tmap_buffer, &a_tmap, 0, sizeof(a_tmap)));
  RT_CHECK(vx_copy_to_dev(b_tmap_buffer, &b_tmap, 0, sizeof(b_tmap)));
  RT_CHECK(vx_copy_to_dev(out_buffer, out_init.data(), 0, out_init.size() * sizeof(float)));

  kernel_arg.grid_dim[0] = 1;
  kernel_arg.grid_dim[1] = 1;
  kernel_arg.grid_dim[2] = 1;
  kernel_arg.block_dim[0] = NUM_THREADS;
  kernel_arg.block_dim[1] = 1;
  kernel_arg.block_dim[2] = 1;
  kernel_arg.a_tmap_addr = a_tmap_addr;
  kernel_arg.b_tmap_addr = b_tmap_addr;
  kernel_arg.out_addr = out_addr;

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  std::vector<float> got(M_DIM * N_DIM);
  RT_CHECK(vx_copy_from_dev(got.data(), out_buffer, 0, got.size() * sizeof(float)));
  cleanup();

  int errors = 0;
  double sum_abs = 0.0;
  double sum_sq = 0.0;
  double max_abs = 0.0;
  for (uint32_t i = 0; i < got.size(); ++i) {
    double diff = std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
    sum_abs += diff;
    sum_sq += diff * diff;
    if (diff > max_abs) {
      max_abs = diff;
    }
    if (diff > 0.01f) {
      if (errors < 8) {
        std::cout << "mismatch[" << i << "]: got=" << got[i]
                  << " ref=" << ref[i] << " diff=" << diff << std::endl;
      }
      ++errors;
    }
  }

  double mean_abs = sum_abs / static_cast<double>(got.size());
  double rms = std::sqrt(sum_sq / static_cast<double>(got.size()));
  std::cout << "ERROR_METRICS: max_abs_error=" << max_abs
            << ", mean_abs_error=" << mean_abs
            << ", rms_error=" << rms << std::endl;

  if (errors == 0) {
    std::cout << "PASSED! mbarrier-async tcgen05 MMA (permutation A) matched reference" << std::endl;
    return 0;
  }
  std::cout << "FAILED: " << errors << " mismatches" << std::endl;
  return 1;
}
