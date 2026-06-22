`timescale 1ns/1ns
`include "define.v"

module to_intype_con #(
    parameter BANK_NUM   = 8,
    parameter BANK_WIDTH = 72,
    parameter DEPTH      = 8,
    parameter BUS_WIDTH  = 512,
    parameter SHAPE_M    = 8,
    parameter SHAPE_K    = 8,
    parameter META_WIDTH = SHAPE_M*(SHAPE_K/4)*`SPARSE_META_BITS_PER_GROUP
)(
    input clk,
    input rst_n,
    input [1:0] a_dtype,
    input [1:0] b_dtype,
    input rd_start_i,
    input [BUS_WIDTH-1:0] a_in_data,
    input a_in_valid_i,
    output a_in_ready_o,
    input [2:0] a_sparse_mode_i,
    input [META_WIDTH-1:0] a_meta_i,
    input a_meta_valid_i,
    input [BUS_WIDTH-1:0] b_in_data,
    input b_in_valid_i,
    output b_in_ready_o,
    input out_ready_i,
    output [BANK_NUM*BANK_WIDTH-1:0] a_out_data,
    output [META_WIDTH-1:0] a_out_meta,
    output [2:0] a_out_sparse_mode,
    output a_out_meta_valid,
    output [BANK_NUM*BANK_WIDTH-1:0] b_out_data,
    output out_valid_o
);

wire a_valid;
wire b_valid;
wire a_meta_valid_local;

matrix_input_buffer_sparse #(
    .BANK_NUM(BANK_NUM),
    .BANK_WIDTH(BANK_WIDTH),
    .DEPTH(DEPTH),
    .BUS_WIDTH(BUS_WIDTH),
    .META_WIDTH(META_WIDTH)
) u_a_buffer (
    .clk(clk),
    .rst_n(rst_n),
    .dtype(a_dtype),
    .rd_start_i(rd_start_i),
    .in_data(a_in_data),
    .transpose_en(1'b0),
    .in_valid_i(a_in_valid_i),
    .out_ready_i(out_ready_i),
    .sparse_mode_i(a_sparse_mode_i),
    .meta_in_data(a_meta_i),
    .meta_in_valid_i(a_meta_valid_i),
    .out_data(a_out_data),
    .out_meta_o(a_out_meta),
    .out_sparse_mode_o(a_out_sparse_mode),
    .out_meta_valid_o(a_meta_valid_local),
    .out_valid_o(a_valid),
    .in_ready_o(a_in_ready_o)
);

matrix_input_buffer #(
    .BANK_NUM(BANK_NUM),
    .BANK_WIDTH(BANK_WIDTH),
    .DEPTH(DEPTH),
    .BUS_WIDTH(BUS_WIDTH)
) u_b_buffer (
    .clk(clk),
    .rst_n(rst_n),
    .dtype(b_dtype),
    .rd_start_i(rd_start_i),
    .in_data(b_in_data),
    .transpose_en(1'b1),
    .in_valid_i(b_in_valid_i),
    .out_ready_i(out_ready_i),
    .out_data(b_out_data),
    .out_valid_o(b_valid),
    .in_ready_o(b_in_ready_o)
);

assign a_out_meta_valid = a_meta_valid_local & a_valid;
assign out_valid_o = a_valid & b_valid;

endmodule

