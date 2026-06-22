`timescale 1ns/1ns
`include "define.v"

module tc_mul_dual_path #(
    parameter SHAPE_M = 8,
    parameter SHAPE_K = 8,
    parameter SHAPE_N = 8,
    parameter ELEM_WIDTH = `XLEN_FP9E5M3
)(
    input  [2:0] sparse_mode_sel_i,
    input  [SHAPE_M*SHAPE_K*ELEM_WIDTH-1:0] dense_a_i,
    input  [SHAPE_M*SHAPE_K*ELEM_WIDTH-1:0] sparse_a_i,
    input  [SHAPE_N*SHAPE_K*ELEM_WIDTH-1:0] dense_b_i,
    output reg [SHAPE_M*SHAPE_K*ELEM_WIDTH-1:0] a_o,
    output reg [SHAPE_N*SHAPE_K*ELEM_WIDTH-1:0] b_o
);

always @(*) begin
    a_o = (sparse_mode_sel_i == `SPARSE_MODE_DENSE) ? dense_a_i : sparse_a_i;
    b_o = dense_b_i;
end

endmodule

