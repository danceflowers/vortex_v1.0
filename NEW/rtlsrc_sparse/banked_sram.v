`timescale 1ns/1ns

module banked_sram #(
    parameter BANK_NUM   = 8,
    parameter BANK_WIDTH = 72,
    parameter DEPTH      = 8
)(
    input                                clk,
    input                                rst_n,
    input      [1:0]                     wr_en,
    input      [$clog2(DEPTH)-1:0]       wr_addr,
    input      [BANK_NUM*BANK_WIDTH-1:0] wr_data,
    input      [1:0]                     rd_en,
    input      [$clog2(DEPTH)-1:0]       rd_addr,
    output reg [BANK_NUM*BANK_WIDTH-1:0] rd_data,
    output reg                           rd_valid
);

localparam TOTAL_W = BANK_NUM*BANK_WIDTH;
localparam HALF_W  = TOTAL_W/2;

reg [TOTAL_W-1:0] mem [0:DEPTH-1];
integer i;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < DEPTH; i = i + 1)
            mem[i] <= {TOTAL_W{1'b0}};
    end else begin
        case (wr_en)
            2'b11: mem[wr_addr] <= wr_data;
            2'b01: mem[wr_addr][HALF_W-1:0] <= wr_data[HALF_W-1:0];
            2'b10: mem[wr_addr][TOTAL_W-1:HALF_W] <= wr_data[TOTAL_W-1:HALF_W];
            default: mem[wr_addr] <= mem[wr_addr];
        endcase
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rd_data  <= {TOTAL_W{1'b0}};
        rd_valid <= 1'b0;
    end else begin
        rd_valid <= |rd_en;
        if (|rd_en)
            rd_data <= mem[rd_addr];
    end
end

endmodule

