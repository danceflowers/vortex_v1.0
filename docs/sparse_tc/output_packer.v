`timescale 1ns / 1ps

module output_packer #(
    parameter integer DATA_WIDTH = 512,
    parameter integer META_WIDTH = 64
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire [DATA_WIDTH-1:0]  raw_c_data_i,
    input  wire                   in_valid_i,
    output reg  [DATA_WIDTH-1:0]  out_data_o,
    output reg  [META_WIDTH-1:0]  out_meta_o,
    output reg  [2:0]             out_sparse_mode_o,
    output reg                    out_valid_o
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_data_o        <= {DATA_WIDTH{1'b0}};
            out_meta_o        <= {META_WIDTH{1'b0}};
            out_sparse_mode_o <= 3'b000;
            out_valid_o       <= 1'b0;
        end else begin
            out_data_o        <= raw_c_data_i;
            out_meta_o        <= {META_WIDTH{1'b0}};
            out_sparse_mode_o <= 3'b000;
            out_valid_o       <= in_valid_i;
        end
    end

endmodule