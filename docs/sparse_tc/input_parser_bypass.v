`timescale 1ns / 1ps

module input_parser_bypass #(
    parameter integer DATA_WIDTH = 512,
    parameter integer META_WIDTH = 64
) (
    input  wire [DATA_WIDTH-1:0] upstream_data_i,
    input  wire [META_WIDTH-1:0] upstream_meta_i,
    input  wire [2:0]            sparse_mode_sel_i,
    input  wire                  a_is_compressed_i,
    input  wire                  meta_valid_i,

    output wire [DATA_WIDTH-1:0] a_data_o,
    output wire [META_WIDTH-1:0] meta_data_o,
    output wire [2:0]            sparse_mode_sel_o,
    output wire                  bypass_en_o
);

    wire use_sparse_bypass = (sparse_mode_sel_i != 3'b000) &&
                             a_is_compressed_i &&
                             meta_valid_i;

    assign a_data_o          = upstream_data_i;
    assign meta_data_o       = use_sparse_bypass ? upstream_meta_i : {META_WIDTH{1'b0}};
    assign sparse_mode_sel_o = sparse_mode_sel_i;
    assign bypass_en_o       = use_sparse_bypass;

endmodule