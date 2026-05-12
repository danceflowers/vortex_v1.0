#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <vortex.h>

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
vx_buffer_h out_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

void cleanup() {
  if (device) {
    if (out_buffer)  vx_mem_free(out_buffer);
    if (krnl_buffer) vx_mem_free(krnl_buffer);
    if (args_buffer) vx_mem_free(args_buffer);
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

  uint64_t num_threads = 0;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));
  if (num_threads != NUM_THREADS) {
    std::cout << "Error: device warp size (" << num_threads
              << ") must match NUM_THREADS=" << NUM_THREADS << std::endl;
    cleanup();
    return -1;
  }

  size_t bytes = NUM_THREADS * sizeof(uint32_t);
  std::vector<uint32_t> sentinel(NUM_THREADS, 0xDEADBEEFu);

  RT_CHECK(vx_mem_alloc(device, bytes, VX_MEM_WRITE, &out_buffer));
  uint64_t out_addr = 0;
  RT_CHECK(vx_mem_address(out_buffer, &out_addr));
  RT_CHECK(vx_copy_to_dev(out_buffer, sentinel.data(), 0, bytes));

  kernel_arg.grid_dim[0] = 1;
  kernel_arg.grid_dim[1] = 1;
  kernel_arg.block_dim[0] = NUM_THREADS;
  kernel_arg.block_dim[1] = 1;
  kernel_arg.out_addr = out_addr;

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  std::vector<uint32_t> observed(NUM_THREADS);
  RT_CHECK(vx_copy_from_dev(observed.data(), out_buffer, 0, bytes));

  cleanup();

  int errors = 0;
  for (uint32_t lane = 0; lane < NUM_THREADS; ++lane) {
    uint32_t expected = 0xCAFE0000u | (lane & 0xFFFFu);
    if (observed[lane] != expected) {
      if (errors < 8) {
        std::cout << "mismatch lane[" << lane << "]: actual=0x" << std::hex
                  << observed[lane] << " expected=0x" << expected << std::dec
                  << std::endl;
      }
      ++errors;
    }
  }

  if (errors == 0) {
    std::cout << "PASSED! tcu_st->tcu_ld round-trip verified across "
              << NUM_THREADS << " lanes" << std::endl;
    return 0;
  }
  std::cout << "FAILED: " << errors << " mismatches" << std::endl;
  return 1;
}
