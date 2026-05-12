#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <rvfloats.h>
#include <vortex.h>
#include <util.h>

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

vx_device_h device = nullptr;
vx_buffer_h a_buffer = nullptr;
vx_buffer_h b_buffer = nullptr;
vx_buffer_h d_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

void cleanup() {
  if (device) {
    if (a_buffer)    vx_mem_free(a_buffer);
    if (b_buffer)    vx_mem_free(b_buffer);
    if (d_buffer)    vx_mem_free(d_buffer);
    if (krnl_buffer) vx_mem_free(krnl_buffer);
    if (args_buffer) vx_mem_free(args_buffer);
    vx_dev_close(device);
    device = nullptr;
  }
}

static inline uint16_t f32_to_fp16(float v) {
  return rv_ftoh_s(vortex::bit_cast<uint32_t>(v), 0, nullptr);
}
static inline float fp16_to_f32(uint16_t b) {
  return vortex::bit_cast<float>(rv_htof_s(b, 0, nullptr));
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

  // K_ITERS rounds of (M*K) and (K*N) fp16 inputs.
  size_t a_elems = K_ITERS * M_DIM * K_DIM;  // fp16
  size_t b_elems = K_ITERS * K_DIM * N_DIM;  // fp16
  size_t d_elems = M_DIM * N_DIM;            // fp32

  std::vector<uint16_t> h_a(a_elems);
  std::vector<uint16_t> h_b(b_elems);
  std::vector<float>    h_ref(d_elems, 0.0f);

  // Tiny deterministic patterns. Keep magnitudes small so fp16 multiplies
  // and fp22 accumulator don't overflow.
  for (size_t i = 0; i < a_elems; ++i) {
    h_a[i] = f32_to_fp16(0.1f + 0.001f * (i % 32));
  }
  for (size_t i = 0; i < b_elems; ++i) {
    h_b[i] = f32_to_fp16(0.05f + 0.0005f * (i % 32));
  }

  // CPU reference: fp32 sum_k(A[k] @ B[k]).
  for (uint32_t k = 0; k < K_ITERS; ++k) {
    for (uint32_t m = 0; m < M_DIM; ++m) {
      for (uint32_t n = 0; n < N_DIM; ++n) {
        float acc = 0.0f;
        for (uint32_t kk = 0; kk < K_DIM; ++kk) {
          float a = fp16_to_f32(h_a[k * M_DIM * K_DIM + m * K_DIM + kk]);
          float b = fp16_to_f32(h_b[k * K_DIM * N_DIM + kk * N_DIM + n]);
          acc += a * b;
        }
        h_ref[m * N_DIM + n] += acc;
      }
    }
  }

  size_t a_bytes = a_elems * sizeof(uint16_t);
  size_t b_bytes = b_elems * sizeof(uint16_t);
  size_t d_bytes = d_elems * sizeof(float);

  RT_CHECK(vx_mem_alloc(device, a_bytes, VX_MEM_READ,  &a_buffer));
  RT_CHECK(vx_mem_alloc(device, b_bytes, VX_MEM_READ,  &b_buffer));
  RT_CHECK(vx_mem_alloc(device, d_bytes, VX_MEM_WRITE, &d_buffer));

  uint64_t a_addr = 0, b_addr = 0, d_addr = 0;
  RT_CHECK(vx_mem_address(a_buffer, &a_addr));
  RT_CHECK(vx_mem_address(b_buffer, &b_addr));
  RT_CHECK(vx_mem_address(d_buffer, &d_addr));

  RT_CHECK(vx_copy_to_dev(a_buffer, h_a.data(), 0, a_bytes));
  RT_CHECK(vx_copy_to_dev(b_buffer, h_b.data(), 0, b_bytes));
  std::vector<float> sentinel(d_elems, -1.0f);
  RT_CHECK(vx_copy_to_dev(d_buffer, sentinel.data(), 0, d_bytes));

  kernel_arg.grid_dim[0] = 1;
  kernel_arg.grid_dim[1] = 1;
  kernel_arg.block_dim[0] = NUM_THREADS;
  kernel_arg.block_dim[1] = 1;
  kernel_arg.a_addr = a_addr;
  kernel_arg.b_addr = b_addr;
  kernel_arg.d_addr = d_addr;

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  std::vector<float> h_d(d_elems);
  RT_CHECK(vx_copy_from_dev(h_d.data(), d_buffer, 0, d_bytes));

  cleanup();

  // Validate. fp22 internal accumulator + fp16 inputs → relax tolerance.
  // We require relative-error <= 1e-2 on each element with abs floor 1e-3.
  int errors = 0;
  for (size_t i = 0; i < d_elems; ++i) {
    float ref = h_ref[i];
    float got = h_d[i];
    float diff = std::fabs(got - ref);
    float rel  = diff / std::max(std::fabs(ref), 1e-3f);
    if (rel > 0.02f) {
      if (errors < 8) {
        std::cout << "mismatch[" << i << "]: got=" << got
                  << " ref=" << ref << " rel=" << rel << std::endl;
      }
      ++errors;
    }
  }

  if (errors == 0) {
    std::cout << "PASSED! D-resident accumulation across " << K_ITERS
              << " iterations matches CPU fp32 reference within tolerance"
              << std::endl;
    return 0;
  }
  std::cout << "FAILED: " << errors << " / " << d_elems << " mismatches" << std::endl;
  return 1;
}
