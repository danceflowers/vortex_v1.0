`ifndef OTC_DEFINE_V
`define OTC_DEFINE_V

`define MATRIX_BUS_WIDTH 512

`define XLEN_FP8      8
`define XLEN_FP9E5M3  9
`define XLEN_FP16     16
`define XLEN_FP22     22

`define SPARSE_MODE_DENSE 3'b000
`define SPARSE_MODE_2TO4  3'b001
`define SPARSE_MODE_1TO4  3'b010
`define SPARSE_META_BITS_PER_GROUP 8

`define NUM_THREAD 32
`define NUM_WARP   8
`define DEPTH_WARP 3
`define REGIDX_WIDTH 5
`define REGEXT_WIDTH 3

`endif

