`timescale 1ns/1ns
`include "define.v"

module matrix_input_buffer #(
    parameter BANK_NUM   = 8,
    parameter BANK_WIDTH = 72,
    parameter DEPTH      = 8,
    parameter BUS_WIDTH  = 512
)(
    input clk,
    input rst_n,
    input [1:0] dtype,
    input rd_start_i,
    input [BUS_WIDTH-1:0] in_data,
    input transpose_en,
    input in_valid_i,
    input out_ready_i,
    output reg [BANK_NUM * BANK_WIDTH - 1:0] out_data,
    output reg out_valid_o,
    output in_ready_o
);

wire [BANK_NUM * BANK_WIDTH - 1:0] conv_data;
wire conv_valid;
wire conv_ready;
reg  [BANK_NUM * BANK_WIDTH - 1:0] stored_data;
reg stored_valid;
reg pending_launch;
reg  [BANK_NUM * BANK_WIDTH - 1:0] transposed_data;
integer r, c;

fp_to_fp9 #(
    .MATRIX_BUS_WIDTH(BUS_WIDTH),
    .LATENCY(1)
) u_fp_to_fp9 (
    .clk(clk),
    .rst_n(rst_n),
    .in_i(in_data),
    .dtype_i(dtype),
    .in_valid_i(in_valid_i),
    .out_ready_i(1'b1),
    .out_o(conv_data),
    .out_valid_o(conv_valid),
    .in_ready_o(conv_ready)
);

always @(*) begin
    transposed_data = stored_data;
    for (r = 0; r < 8; r = r + 1)
        for (c = 0; c < 8; c = c + 1)
            transposed_data[(r*8+c)*9 +: 9] = stored_data[(c*8+r)*9 +: 9];
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        stored_data <= {(BANK_NUM * BANK_WIDTH){1'b0}};
        stored_valid <= 1'b0;
        pending_launch <= 1'b0;
        out_data <= {(BANK_NUM * BANK_WIDTH){1'b0}};
        out_valid_o <= 1'b0;
    end else begin
        if (rd_start_i)
            pending_launch <= 1'b1;

        if (conv_valid) begin
            stored_data <= conv_data;
            stored_valid <= 1'b1;
        end

        if (pending_launch && stored_valid && out_ready_i) begin
            out_data <= transpose_en ? transposed_data : stored_data;
            out_valid_o <= 1'b1;
            pending_launch <= 1'b0;
            stored_valid <= 1'b0;
        end else begin
            out_valid_o <= 1'b0;
        end
    end
end

assign in_ready_o = conv_ready;

endmodule

