`timescale 1ns/1ns
`include "define.v"

module to_next_con #(
    parameter BANK_NUM   = 8,
    parameter BANK_WIDTH = 64,
    parameter DEPTH      = 16,
    parameter BUS_WIDTH  = 512,
    parameter OUT_WIDTH  = 1408
)(
    input clk,
    input rst_n,
    input [1:0] dtype,
    input rd_start_i,
    input [BUS_WIDTH-1:0] in_data,
    input in_valid_i,
    output in_ready_o,
    input out_ready_i,
    output reg [OUT_WIDTH-1:0] out_data,
    output reg out_valid_o
);

reg [OUT_WIDTH-1:0] stored_data;
reg stored_valid;
reg pending_launch;
integer i;

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

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        stored_data <= {OUT_WIDTH{1'b0}};
        stored_valid <= 1'b0;
        pending_launch <= 1'b0;
        out_data <= {OUT_WIDTH{1'b0}};
        out_valid_o <= 1'b0;
    end else begin
        if (rd_start_i)
            pending_launch <= 1'b1;

        if (in_valid_i) begin
            stored_data <= {OUT_WIDTH{1'b0}};
            case (dtype)
                2'b10: for (i = 0; i < 32; i = i + 1) stored_data[i*22 +: 22] <= sx16_to_fp22(in_data[i*16 +: 16]);
                2'b11: for (i = 0; i < 16; i = i + 1) stored_data[i*22 +: 22] <= sx32_to_fp22_clip(in_data[i*32 +: 32]);
                default: for (i = 0; i < 64; i = i + 1) stored_data[i*22 +: 22] <= sx8_to_fp22(in_data[i*8 +: 8]);
            endcase
            stored_valid <= 1'b1;
        end

        if (pending_launch && out_ready_i && stored_valid) begin
            out_data <= stored_data;
            out_valid_o <= 1'b1;
            pending_launch <= 1'b0;
            stored_valid <= 1'b0;
        end else begin
            out_valid_o <= 1'b0;
        end
    end
end

assign in_ready_o = 1'b1;

endmodule

