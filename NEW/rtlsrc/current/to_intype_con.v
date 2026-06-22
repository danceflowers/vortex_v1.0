module to_intype_con #(
    parameter BANK_NUM   = 8,
    parameter BANK_WIDTH = 72,
    parameter DEPTH      = 8,
    parameter BUS_WIDTH  = 512
)(
    input clk,
    input rst_n,

    input [1:0] a_dtype,  //10:fp16,01:fp8(e4m3),00:fp8(e5m2)
    input [1:0] b_dtype,  //10:fp16,01:fp8(e4m3),00:fp8(e5m2)
    input rd_start_i,

    // A operand input
    input  [BUS_WIDTH-1:0] a_in_data,
    input                  a_in_valid_i,
    output                 a_in_ready_o,

    // B operand input
    input  [BUS_WIDTH-1:0] b_in_data,
    input                  b_in_valid_i,
    output                 b_in_ready_o,

    input out_ready_i,

    // A operand output
    output [BANK_NUM*BANK_WIDTH-1:0] a_out_data,

    // B operand output
    output [BANK_NUM*BANK_WIDTH-1:0] b_out_data,

    // merged valid
    output out_valid_o
);

//////////////////////////////////////////////////
// internal signals
//////////////////////////////////////////////////

wire a_valid;
wire b_valid;

//////////////////////////////////////////////////
// A matrix buffer
//////////////////////////////////////////////////

matrix_input_buffer #(
    .BANK_NUM(BANK_NUM),
    .BANK_WIDTH(BANK_WIDTH),
    .DEPTH(DEPTH),
    .BUS_WIDTH(BUS_WIDTH)
) u_a_buffer (

    .clk(clk),
    .rst_n(rst_n),

    .dtype(a_dtype),
    .rd_start_i(rd_start_i),

    .in_data(a_in_data),
    .transpose_en(1'b0),   // A matrix 不转置

    .in_valid_i(a_in_valid_i),
    .out_ready_i(out_ready_i),

    .out_data(a_out_data),
    .out_valid_o(a_valid),
    .in_ready_o(a_in_ready_o)

);

//////////////////////////////////////////////////
// B matrix buffer
//////////////////////////////////////////////////

matrix_input_buffer #(
    .BANK_NUM(BANK_NUM),
    .BANK_WIDTH(BANK_WIDTH),
    .DEPTH(DEPTH),
    .BUS_WIDTH(BUS_WIDTH)
) u_b_buffer (

    .clk(clk),
    .rst_n(rst_n),

    .dtype(b_dtype),
    .rd_start_i(rd_start_i),

    .in_data(b_in_data),
    .transpose_en(1'b1),   // B matrix 转置

    .in_valid_i(b_in_valid_i),
    .out_ready_i(out_ready_i),

    .out_data(b_out_data),
    .out_valid_o(b_valid),
    .in_ready_o(b_in_ready_o)

);

//////////////////////////////////////////////////
// valid merge
//////////////////////////////////////////////////

assign out_valid_o = a_valid & b_valid;

endmodule