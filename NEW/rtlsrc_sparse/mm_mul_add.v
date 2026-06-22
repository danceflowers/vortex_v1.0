`timescale 1ns/1ns
`include "define.v"

module mm_mul_add #(
    parameter VL          = `NUM_THREAD,
    parameter SHAPE_M     = 8,
    parameter SHAPE_K     = 8,
    parameter SHAPE_N     = 8,
    parameter EXPWIDTH    = 5,
    parameter PRECISION   = 4,
    parameter SUM_OUT_EXP = 8,
    parameter SUM_OUT_MAN = 13
)(
    input clk,
    input rst_n,
    input [SHAPE_M*SHAPE_K*`XLEN_FP9E5M3-1:0]  a_i,
    input [SHAPE_N*SHAPE_K*`XLEN_FP9E5M3-1:0]  b_i,
    input [SHAPE_M*SHAPE_N*`XLEN_FP22-1:0]     c_i,
    input [VL*3-1:0] rm_i,
    input in_valid_i,
    input out_ready_i,
    output in_ready_o,
    output reg out_valid_o,
    output reg [`MATRIX_BUS_WIDTH*2-1:0] result_o,
    output reg [(SHAPE_M * SHAPE_N * 5) - 1 : 0] fflags_o
);

integer i, j, k;
integer acc;
reg signed [8:0] a_lane;
reg signed [8:0] b_lane;
reg signed [21:0] c_lane;
reg [`MATRIX_BUS_WIDTH*2-1:0] result_next;

function signed [31:0] sx9;
    input [8:0] x;
    begin
        sx9 = {{23{x[8]}}, x};
    end
endfunction

function signed [31:0] sx22;
    input [21:0] x;
    begin
        sx22 = {{10{x[21]}}, x};
    end
endfunction

always @(*) begin
    result_next = {(`MATRIX_BUS_WIDTH*2){1'b0}};
    for (i = 0; i < SHAPE_M; i = i + 1) begin
        for (j = 0; j < SHAPE_N; j = j + 1) begin
            c_lane = c_i[((i*SHAPE_N+j)*`XLEN_FP22) +: `XLEN_FP22];
            acc = sx22(c_lane);
            for (k = 0; k < SHAPE_K; k = k + 1) begin
                a_lane = a_i[((i*SHAPE_K+k)*`XLEN_FP9E5M3) +: `XLEN_FP9E5M3];
                b_lane = b_i[((j*SHAPE_K+k)*`XLEN_FP9E5M3) +: `XLEN_FP9E5M3];
                acc = acc + sx9(a_lane) * sx9(b_lane);
            end
            result_next[((i*SHAPE_N+j)*16) +: 16] = acc[15:0];
        end
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        result_o <= {(`MATRIX_BUS_WIDTH*2){1'b0}};
        fflags_o <= {(SHAPE_M * SHAPE_N * 5){1'b0}};
        out_valid_o <= 1'b0;
    end else begin
        if (in_valid_i && out_ready_i) begin
            result_o <= result_next;
            fflags_o <= {(SHAPE_M * SHAPE_N * 5){1'b0}};
            out_valid_o <= 1'b1;
        end else begin
            out_valid_o <= 1'b0;
        end
    end
end

assign in_ready_o = 1'b1;

endmodule

