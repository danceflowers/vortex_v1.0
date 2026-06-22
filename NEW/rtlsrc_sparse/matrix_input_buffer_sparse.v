`timescale 1ns/1ns
`include "define.v"

module matrix_input_buffer_sparse #(
    parameter BANK_NUM   = 8,
    parameter BANK_WIDTH = 72,
    parameter DEPTH      = 8,
    parameter BUS_WIDTH  = 512,
    parameter META_WIDTH = 128
)(
    input clk,
    input rst_n,
    input [1:0] dtype,
    input rd_start_i,
    input [BUS_WIDTH-1:0] in_data,
    input transpose_en,
    input in_valid_i,
    input out_ready_i,
    input [2:0] sparse_mode_i,
    input [META_WIDTH-1:0] meta_in_data,
    input meta_in_valid_i,
    output reg [BANK_NUM*BANK_WIDTH-1:0] out_data,
    output reg [META_WIDTH-1:0] out_meta_o,
    output reg [2:0] out_sparse_mode_o,
    output reg out_meta_valid_o,
    output reg out_valid_o,
    output in_ready_o
);

wire [BANK_NUM * BANK_WIDTH - 1:0] conv_data;
wire conv_valid;
wire conv_ready;
reg  [BANK_NUM * BANK_WIDTH - 1:0] stored_data;
reg  [BANK_NUM * BANK_WIDTH - 1:0] transposed_data;
reg  [META_WIDTH-1:0] stored_meta;
reg  [2:0] stored_sparse_mode;
reg  stored_meta_valid;
reg  stored_valid;
reg  pending_launch;

// Latch sideband control at the same time the raw A payload is accepted.
// fp_to_fp9 adds one registered stage, so sparse_mode/meta cannot be sampled
// later when conv_valid asserts, otherwise the testbench's one-cycle control
// pulse has already been deasserted and sparse requests collapse back to dense.
reg  [META_WIDTH-1:0] pending_meta;
reg  [2:0] pending_sparse_mode;
reg  pending_meta_valid;

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
        stored_meta <= {META_WIDTH{1'b0}};
        stored_sparse_mode <= `SPARSE_MODE_DENSE;
        stored_meta_valid <= 1'b0;
        stored_valid <= 1'b0;
        pending_launch <= 1'b0;
        pending_meta <= {META_WIDTH{1'b0}};
        pending_sparse_mode <= `SPARSE_MODE_DENSE;
        pending_meta_valid <= 1'b0;
        out_data <= {(BANK_NUM * BANK_WIDTH){1'b0}};
        out_meta_o <= {META_WIDTH{1'b0}};
        out_sparse_mode_o <= `SPARSE_MODE_DENSE;
        out_meta_valid_o <= 1'b0;
        out_valid_o <= 1'b0;
    end else begin
        if (rd_start_i)
            pending_launch <= 1'b1;

        // Capture sparse sideband in the same cycle as the raw input beat.
        if (in_valid_i && conv_ready) begin
            pending_meta <= meta_in_data;
            pending_sparse_mode <= sparse_mode_i;
            pending_meta_valid <= meta_in_valid_i;
        end

        // When the converted payload becomes valid, combine it with the
        // previously latched sideband.
        if (conv_valid) begin
            stored_data <= conv_data;
            stored_meta <= pending_meta;
            stored_sparse_mode <= pending_sparse_mode;
            stored_meta_valid <= pending_meta_valid;
            stored_valid <= 1'b1;
        end

        if (pending_launch && stored_valid && out_ready_i) begin
            out_data <= transpose_en ? transposed_data : stored_data;
            out_meta_o <= stored_meta;
            out_sparse_mode_o <= stored_sparse_mode;
            out_meta_valid_o <= stored_meta_valid;
            out_valid_o <= 1'b1;
            pending_launch <= 1'b0;
            stored_valid <= 1'b0;
        end else begin
            out_valid_o <= 1'b0;
            out_meta_valid_o <= 1'b0;
        end
    end
end

assign in_ready_o = conv_ready;

endmodule

