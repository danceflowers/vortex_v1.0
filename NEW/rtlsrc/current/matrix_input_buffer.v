module matrix_input_buffer #(
    parameter BANK_NUM   = 8,
    parameter BANK_WIDTH = 72,
    parameter DEPTH      = 8,
    parameter BUS_WIDTH   = 512
)(
    input clk,
    input rst_n,

    input [1:0] dtype, //10:fp16,01:fp8(e4m3),00:fp8(e5m2)
    input rd_start_i,   // trigger read
    input [BUS_WIDTH-1:0] in_data,
    input transpose_en,    //amem :0; bmem : 1
    input in_valid_i,      //input data valid
    input out_ready_i,     //next stage ready

    output [BANK_NUM * BANK_WIDTH - 1:0] out_data,
    output     out_valid_o,    //output data valid
    output     in_ready_o      //ready to accept input data
    );
//////////////////////////////////////////////////
// dtype decode
//////////////////////////////////////////////////

wire fp16 = dtype==2'b10;
wire fp8e4 = dtype==2'b01;
wire fp8e5 = dtype==2'b00;

wire fp8 = dtype[0];
//////////////////////////////////////////////////
// to_intype_con
//////////////////////////////////////////////////

wire [BANK_NUM * BANK_WIDTH - 1:0] conv_data;
wire conv_valid;

fp_to_fp9 #(
    .MATRIX_BUS_WIDTH(BUS_WIDTH),
    .LATENCY(1)
) u_fp_to_fp9(
    .clk(clk),
    .rst_n(rst_n),

    .in_i(in_data),
    .dtype_i(dtype),

    .in_valid_i(in_valid_i),
    .out_ready_i(1'b1),

    .out_o(conv_data),
    .out_valid_o(conv_valid),
    .in_ready_o(in_ready_o)  
);

////////////////////////////////////////////////////////////
// write control
////////////////////////////////////////////////////////////

reg [$clog2(DEPTH)-1:0] wr_addr;
reg half_wr;

always @(posedge clk or negedge rst_n) begin
if(!rst_n) begin
    wr_addr <= 0;
    half_wr <= 0;
end
else if(conv_valid) begin  //输入有效

    if(fp16) begin
        if(half_wr) begin
            wr_addr <= wr_addr + 1;
            half_wr <= 0;
        end
        else begin
            half_wr <= 1;
        end
    end
    else begin
        wr_addr <= wr_addr + 1;
    end

end
end

////////////////////////////////////////////////////////////
// write enable
////////////////////////////////////////////////////////////
reg [1:0] wr_en;

always @(*) begin
wr_en = 0;
if(conv_valid) begin
    if(fp16) begin
        wr_en = half_wr ? 2'b10 : 2'b01;
    end
    else begin
        wr_en = 2'b11;
    end
end
else begin
    wr_en = 2'b00;
end
end
////////////////////////////////////////////////////////////
// write data mux
////////////////////////////////////////////////////////////
reg [BANK_NUM * BANK_WIDTH - 1:0] wr_data;

always @(*) begin
    if(fp16) begin
        wr_data = half_wr ? {conv_data[287:0],288'd0} : {288'd0,conv_data[287:0]};   //todo:这里没有参数化
    end
    else
        wr_data = conv_data;
    end

////////////////////////////////////////////////////////////
// read control
////////////////////////////////////////////////////////////
reg [2:0] rd_addr;
//reg half_rd;

always @(posedge clk or negedge rst_n) begin

if(!rst_n) begin
    rd_addr <= 0;
//    half_rd <= 0;
end
else if(rd_start_i & out_ready_i) begin  //开始读，且下游准备好接受数据
    rd_addr <= rd_addr + 1;
end
end

////////////////////////////////////////////////////////////
// read enable
////////////////////////////////////////////////////////////

reg [1:0] rd_en;

always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        rd_en = 0;
    end
    else if(rd_start_i & out_ready_i) begin
        rd_en = 2'b11;
        // if(fp4)
        //     rd_en = half_rd ? 2'b01 : 2'b10;
        // else
        //     rd_en = 2'b11;
    end
else begin
    rd_en = 2'b00;
end

end
////////////////////////////////////////////////////////////
// SRAM instance
////////////////////////////////////////////////////////////

wire [575:0] rd_data;
wire rd_data_valid;
banked_sram #(
.BANK_NUM(BANK_NUM),
.BANK_WIDTH(BANK_WIDTH),
.DEPTH(DEPTH)
) u_banked_sram(

.clk(clk),
.rst_n(rst_n),

.wr_en(wr_en),
.wr_addr(wr_addr),
.wr_data(wr_data),

.rd_en(rd_en),
.rd_addr(rd_addr),

.rd_data(rd_data),
.rd_valid(rd_data_valid)  //读出数据有效

);


wire [575:0] reorder_in;
assign reorder_in = rd_data;
  

////////////////////////////////////////////////////////////
// reorder
////////////////////////////////////////////////////////////

reordering u_reorder(

.in_data(reorder_in),
.transpose_en(transpose_en),
.out_data(out_data)

);     

////////////////////////////////////////////////////////////
// output control
////////////////////////////////////////////////////////////
assign out_valid_o = rd_data_valid;  //读出数据有效即输出有效


endmodule

