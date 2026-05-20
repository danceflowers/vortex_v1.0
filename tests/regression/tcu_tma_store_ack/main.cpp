#include "common.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

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

struct host_tensor_map_t {
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
static_assert(sizeof(host_tensor_map_t) == 128, "tensor_map_t must be 128 B");

static const char* kernel_file = "kernel.vxbin";

vx_device_h device = nullptr;
vx_buffer_h out_buffer = nullptr;
vx_buffer_h tmap_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;

void cleanup() {
  if (device) {
    if (out_buffer) vx_mem_free(out_buffer);
    if (tmap_buffer) vx_mem_free(tmap_buffer);
    if (krnl_buffer) vx_mem_free(krnl_buffer);
    if (args_buffer) vx_mem_free(args_buffer);
    vx_dev_close(device);
    device = nullptr;
  }
}

} // namespace

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

  RT_CHECK(vx_mem_alloc(device, TMA_STORE_BYTES, VX_MEM_READ_WRITE,
                        &out_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(host_tensor_map_t), VX_MEM_READ,
                        &tmap_buffer));

  uint64_t out_addr = 0;
  uint64_t tmap_addr = 0;
  RT_CHECK(vx_mem_address(out_buffer, &out_addr));
  RT_CHECK(vx_mem_address(tmap_buffer, &tmap_addr));

  std::vector<uint32_t> sentinel(TMA_STORE_WORDS, 0xDEADBEEFu);
  RT_CHECK(vx_copy_to_dev(out_buffer, sentinel.data(), 0, TMA_STORE_BYTES));

  host_tensor_map_t tmap = {};
  tmap.global_address = out_addr;
  tmap.box_size[0] = TMA_STORE_WORDS;
  tmap.global_stride[0] = 4;
  tmap.element_strides[0] = 1;
  tmap.element_type = 2;
  tmap.rank = 1;
  RT_CHECK(vx_copy_to_dev(tmap_buffer, &tmap, 0, sizeof(tmap)));

  kernel_arg_t kernel_arg = {};
  kernel_arg.grid_dim[0] = 1;
  kernel_arg.grid_dim[1] = 1;
  kernel_arg.block_dim[0] = NUM_THREADS;
  kernel_arg.block_dim[1] = 1;
  kernel_arg.tensor_map_addr = tmap_addr;
  kernel_arg.out_addr = out_addr;

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  std::vector<uint32_t> observed(TMA_STORE_WORDS, 0);
  RT_CHECK(vx_copy_from_dev(observed.data(), out_buffer, 0, TMA_STORE_BYTES));

  cleanup();

  for (uint32_t i = 0; i < TMA_STORE_WORDS; ++i) {
    uint32_t expected = 0x5A5A0000u | (i & 0xffffu);
    if (observed[i] != expected) {
      std::cout << "FAILED: index=" << i
                << " observed=0x" << std::hex << observed[i]
                << " expected=0x" << expected << std::endl;
      return 1;
    }
  }

  std::cout << "PASSED! TMA store completed via write acknowledgement" << std::endl;
  return 0;
}
