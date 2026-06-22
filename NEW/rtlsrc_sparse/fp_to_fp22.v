`timescale 1ns/1ns
`include "define.v"

module fp_to_fp22 #(
    parameter MATRIX_BUS_WIDTH = `MATRIX_BUS_WIDTH,
    parameter OUT_BUS_WIDTH = 1408
)(
    input  clk,
    input  rst_n,
    input  [MATRIX_BUS_WIDTH-1:0] in_i,
    input  [1:0] dtype_i,
    input  in_valid_i,
    input  out_ready_i,
    output reg [OUT_BUS_WIDTH-1:0] out_o,
    output reg out_valid_o,
    output in_ready_o
);

integer i;
reg [OUT_BUS_WIDTH-1:0] out_next;

function [21:0] sx8_to_fp22;
    input [7:0] x;
    begin
        sx8_to_fp22 = {{14{x[7]}}, x};
    end
endfunction

function [21:0] sx16_to_fp22;
    input [15:0] x;
    begin
        sx16_to_fp22 = {{6{x[15]}}, x};
    end
endfunction

function [21:0] sx32_to_fp22_clip;
    input [31:0] x;
    begin
        sx32_to_fp22_clip = x[21:0];
    end
endfunction

always @(*) begin
    out_next = {OUT_BUS_WIDTH{1'b0}};
    case (dtype_i)
        2'b10: begin
            for (i = 0; i < 32; i = i + 1)
                out_next[i*22 +: 22] = sx16_to_fp22(in_i[i*16 +: 16]);
        end
        2'b11: begin
            for (i = 0; i < 16; i = i + 1)
                out_next[i*22 +: 22] = sx32_to_fp22_clip(in_i[i*32 +: 32]);
        end
        default: begin
            for (i = 0; i < 64; i = i + 1)
                out_next[i*22 +: 22] = sx8_to_fp22(in_i[i*8 +: 8]);
        end
    endcase
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        out_o <= {OUT_BUS_WIDTH{1'b0}};
        out_valid_o <= 1'b0;
    end else begin
        out_o <= out_next;
        out_valid_o <= in_valid_i;
    end
end

assign in_ready_o = out_ready_i;

endmodule

