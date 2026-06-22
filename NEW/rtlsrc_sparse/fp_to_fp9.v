`timescale 1ns/1ns
`include "define.v"

module fp_to_fp9 #(
    parameter MATRIX_BUS_WIDTH = `MATRIX_BUS_WIDTH,
    parameter LATENCY = 1
)(
    input  clk,
    input  rst_n,
    input  [MATRIX_BUS_WIDTH-1:0] in_i,
    input  [1:0] dtype_i,
    input  in_valid_i,
    input  out_ready_i,
    output reg [MATRIX_BUS_WIDTH*9/8-1:0] out_o,
    output reg out_valid_o,
    output in_ready_o
);

localparam FP8_NUM  = MATRIX_BUS_WIDTH / 8;
localparam FP16_NUM = MATRIX_BUS_WIDTH / 16;
integer i;
reg [MATRIX_BUS_WIDTH*9/8-1:0] out_next;

function [8:0] sx8_to_fp9;
    input [7:0] x;
    begin
        sx8_to_fp9 = {x[7], x};
    end
endfunction

function [8:0] fp16_to_fp9_sim;
    input [15:0] x;
    begin
        fp16_to_fp9_sim = {x[15], x[14:7]};
    end
endfunction

always @(*) begin
    out_next = {((MATRIX_BUS_WIDTH*9/8)){1'b0}};
    if (dtype_i == 2'b10) begin
        for (i = 0; i < FP16_NUM; i = i + 1)
            out_next[i*9 +: 9] = fp16_to_fp9_sim(in_i[i*16 +: 16]);
    end else begin
        for (i = 0; i < FP8_NUM; i = i + 1)
            out_next[i*9 +: 9] = sx8_to_fp9(in_i[i*8 +: 8]);
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        out_o <= {((MATRIX_BUS_WIDTH*9/8)){1'b0}};
        out_valid_o <= 1'b0;
    end else begin
        out_o <= out_next;
        out_valid_o <= in_valid_i;
    end
end

assign in_ready_o = out_ready_i;

endmodule

