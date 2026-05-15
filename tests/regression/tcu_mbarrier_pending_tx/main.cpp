#include <cstdint>
#include <cstdlib>
#include <cstring>
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

// Minimal tensor_map_t mirror of sim/simx/tensor/idescriptor.h::tensor_map_t.
// Host fills the bytes the kernel will see at kernel_arg.tensor_map_addr.
struct __attribute__((packed)) host_tensor_map_t {
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
};
static_assert(sizeof(host_tensor_map_t) == 128, "tensor_map_t must be 128 B");

static constexpr uint32_t kElems = 1024;          // u32 elements
static constexpr uint32_t kElemBytes = 4;         // element_type=2 -> U32
static constexpr uint32_t kTileBytes = kElems * kElemBytes;

vx_device_h device = nullptr;
vx_buffer_h input_buffer = nullptr;
vx_buffer_h tmap_buffer = nullptr;
vx_buffer_h out_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

void cleanup() {
  if (device) {
    if (input_buffer) vx_mem_free(input_buffer);
    if (tmap_buffer)  vx_mem_free(tmap_buffer);
    if (out_buffer)   vx_mem_free(out_buffer);
    if (krnl_buffer)  vx_mem_free(krnl_buffer);
    if (args_buffer)  vx_mem_free(args_buffer);
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

  // Source data: u32[kElems] with a recognizable pattern.
  std::vector<uint32_t> h_input(kElems);
  for (uint32_t i = 0; i < kElems; ++i) {
    h_input[i] = 0xCAFE0000u | (i & 0xFFFFu);
  }

  // Allocate DRAM buffers.
  RT_CHECK(vx_mem_alloc(device, kTileBytes,             VX_MEM_READ,  &input_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(host_tensor_map_t), VX_MEM_READ, &tmap_buffer));
  RT_CHECK(vx_mem_alloc(device, sizeof(uint32_t),       VX_MEM_WRITE, &out_buffer));

  uint64_t input_addr = 0, tmap_addr = 0, out_addr = 0;
  RT_CHECK(vx_mem_address(input_buffer, &input_addr));
  RT_CHECK(vx_mem_address(tmap_buffer,  &tmap_addr));
  RT_CHECK(vx_mem_address(out_buffer,   &out_addr));

  RT_CHECK(vx_copy_to_dev(input_buffer, h_input.data(), 0, kTileBytes));

  // Populate the tensor_map_t. rank=1, box_size[0]=kElems, element_type=2 (U32).
  host_tensor_map_t tmap = {};
  tmap.global_address = input_addr;
  tmap.box_size[0]    = kElems;
  tmap.global_stride[0] = kElemBytes;
  tmap.element_strides[0] = 1;
  tmap.element_type   = 2;  // U32 per cpabulk_tensor_load element_type_bytes()
  tmap.rank           = 1;
  RT_CHECK(vx_copy_to_dev(tmap_buffer, &tmap, 0, sizeof(tmap)));

  // Pre-fill out_addr with a sentinel so we can tell if the kernel wrote it.
  uint32_t sentinel = 0xDEADBEEFu;
  RT_CHECK(vx_copy_to_dev(out_buffer, &sentinel, 0, sizeof(sentinel)));

  kernel_arg.grid_dim[0] = 1;
  kernel_arg.grid_dim[1] = 1;
  kernel_arg.block_dim[0] = NUM_THREADS;
  kernel_arg.block_dim[1] = 1;
  kernel_arg.tensor_map_addr = tmap_addr;
  kernel_arg.out_addr        = out_addr;
  kernel_arg.tx_bytes        = kTileBytes;

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  uint32_t observed = 0;
  RT_CHECK(vx_copy_from_dev(&observed, out_buffer, 0, sizeof(observed)));

  cleanup();

  if (observed == h_input[0]) {
    std::cout << "PASSED! (observed first word = 0x" << std::hex << observed
              << " matches DRAM source via LMEM direct path)" << std::endl;
    return 0;
  }

  std::cout << "FAILED: observed=0x" << std::hex << observed
            << " expected=0x" << h_input[0] << std::endl;
  return 1;
}
