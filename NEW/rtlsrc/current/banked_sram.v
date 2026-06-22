module banked_sram #(
    parameter BANK_NUM   = 8,
    parameter BANK_WIDTH = 72,
    parameter DEPTH      = 8
)(
    input clk,
    input rst_n,

    // write
    input  [1:0] wr_en,
    input  [$clog2(DEPTH)-1:0] wr_addr,
    input  [BANK_NUM*BANK_WIDTH-1:0] wr_data,

    // read
    input  [1:0] rd_en,
    input  [$clog2(DEPTH)-1:0] rd_addr,

    output  [BANK_NUM*BANK_WIDTH-1:0] rd_data,
    output reg rd_valid
);

//////////////////////////////////////////////////////
// bank memories
//////////////////////////////////////////////////////

genvar b;

generate
for(b=0;b<BANK_NUM;b=b+1) begin: BANK

    reg [BANK_WIDTH-1:0] mem [0:DEPTH-1];
    reg [BANK_WIDTH-1:0] rdata;

    wire bank_wr_en;
    wire bank_rd_en;

    // bank enable decode
    assign bank_wr_en = (b < BANK_NUM/2) ? wr_en[0] : wr_en[1];
    assign bank_rd_en = (b < BANK_NUM/2) ? rd_en[0] : rd_en[1];

    //////////////////////////////////////////////////////
    // write
    //////////////////////////////////////////////////////

    always @(posedge clk) begin
        if(bank_wr_en)
            mem[wr_addr] <= wr_data[b*BANK_WIDTH +: BANK_WIDTH];
    end

    //////////////////////////////////////////////////////
    // read
    //////////////////////////////////////////////////////

    always @(posedge clk) begin
        if(bank_rd_en)
            rdata <= mem[rd_addr];
    end

    //////////////////////////////////////////////////////
    // output pack
    //////////////////////////////////////////////////////


    assign   rd_data[b*BANK_WIDTH +: BANK_WIDTH] = rdata;


end
endgenerate

//////////////////////////////////////////////////////
// valid
//////////////////////////////////////////////////////

always @(posedge clk or negedge rst_n) begin
    if(!rst_n)
        rd_valid <= 1'b0;
    else
        rd_valid <= |rd_en;
end

endmodule