`timescale 1ns/1ns
`include "define.v"

module input_parser_bypass #(
    parameter BUS_WIDTH  = `MATRIX_BUS_WIDTH,
    parameter META_WIDTH = 128
)(
    input  clk,
    input  rst_n,
    input  [2:0] sparse_mode_sel_i,
    input  a_is_compressed_i,
    input  [META_WIDTH-1:0] meta_i,
    input  meta_valid_i,
    input  [BUS_WIDTH-1:0] a_data_i,
    input  a_valid_i,
    output a_ready_o,
    input  out_ready_i,
    output reg [BUS_WIDTH-1:0] a_data_o,
    output reg [META_WIDTH-1:0] meta_o,
    output reg [2:0] sparse_mode_sel_o,
    output reg meta_valid_o,
    output reg out_valid_o
);

assign a_ready_o = out_ready_i;

always @(*) begin
    a_data_o = a_data_i;
    meta_o = {META_WIDTH{1'b0}};
    sparse_mode_sel_o = `SPARSE_MODE_DENSE;
    meta_valid_o = 1'b0;
    out_valid_o = a_valid_i;

    if ((sparse_mode_sel_i != `SPARSE_MODE_DENSE) && a_is_compressed_i && meta_valid_i) begin
        meta_o = meta_i;
        sparse_mode_sel_o = sparse_mode_sel_i;
        meta_valid_o = 1'b1;
    end
end

endmodule

