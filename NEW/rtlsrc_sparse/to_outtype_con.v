`timescale 1ns/1ns
`include "define.v"

module to_outtype_con #(
    parameter BANK_NUM   = 16,
    parameter BANK_WIDTH = 64,
    parameter DEPTH      = 8,
    parameter IN_BUS_WIDTH  = 1024,
    parameter OUT_BUS_WIDTH = 512
)(
    input clk,
    input rst_n,
    input d_dtype,
    input [IN_BUS_WIDTH-1:0] d_in_data,
    input in_valid_i,
    output in_ready_o,
    input out_ready_i,
    output reg out_valid_o,
    output reg [OUT_BUS_WIDTH-1:0] d_out_data
);

integer i;
reg [OUT_BUS_WIDTH-1:0] out_next;

always @(*) begin
    out_next = {OUT_BUS_WIDTH{1'b0}};
    for (i = 0; i < 64; i = i + 1)
        out_next[i*8 +: 8] = d_in_data[i*16 +: 8];
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        d_out_data <= {OUT_BUS_WIDTH{1'b0}};
        out_valid_o <= 1'b0;
    end else begin
        if (in_valid_i && out_ready_i) begin
            d_out_data <= out_next;
            out_valid_o <= 1'b1;
        end else begin
            out_valid_o <= 1'b0;
        end
    end
end

assign in_ready_o = 1'b1;

endmodule

