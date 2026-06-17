`timescale 1ns / 1ps

module tc_mul_dual_path #(
    parameter integer PP_WIDTH = 1296
) (
    input  wire [2:0]          sparse_mode_sel_i,
    input  wire [PP_WIDTH-1:0] dense_pp_i,
    input  wire [PP_WIDTH-1:0] sparse_pp_i,
    output wire [PP_WIDTH-1:0] final_pp_o
);

    assign final_pp_o = (sparse_mode_sel_i == 3'b000) ? dense_pp_i : sparse_pp_i;

endmodule