module to_outtype_con #(
    parameter BANK_NUM   = 16,
    parameter BANK_WIDTH = 64,
    parameter DEPTH      = 8,
    parameter IN_BUS_WIDTH  = 1408,
    parameter OUT_BUS_WIDTH = 512
)(
    input clk,
    input rst_n,

    input  d_dtype,  //1:fp16,0:fp8
    
    input  [IN_BUS_WIDTH-1:0] d_in_data,
    input  in_valid_i,
    output in_ready_o,

    input out_ready_i,   //外面ready就自己开始读
    output reg out_valid_o,

    output reg [OUT_BUS_WIDTH-1:0] d_out_data
);
//dtype_conv
reg [`XLEN_FP22-1:0] d_in;
reg [7:0]  exp22_raw ;
reg [12:0] mant22_raw ;
reg [3:0]  exp8_out_new;
reg [4:0]  exp16_out_new;
reg [15:0] result_fp16;
reg [7:0] result_fp8;
reg [OUT_BUS_WIDTH*2-1:0] conv_out;
integer i;
always @(*) begin
    for(i=0;i<64;i=i+1) begin
        d_in = d_in_data[i*`XLEN_FP22 +: 22];
        exp22_raw = d_in[20:13];
        mant22_raw = d_in[12:0];

        exp8_out_new = (exp22_raw < 8'd120) ? 4'd0 : (exp22_raw[3:0] - 4'd8);
        exp16_out_new = (exp22_raw < 8'd112) ? 5'd0 :(exp22_raw[4:0] - 5'd16);

        result_fp8 = { d_in[21], exp8_out_new, mant22_raw[12:10] };
        result_fp16 = { d_in[21], exp16_out_new, mant22_raw[12:3] };

        if(d_dtype)  //fp16
            conv_out[i*`XLEN_FP16 +: `XLEN_FP16] = result_fp16;
        else
            conv_out[i*`XLEN_FP8 +: `XLEN_FP8] = result_fp8;
end
    
end


///////////////////////////////////////////////////////////
// write control
////////////////////////////////////////////////////////////
reg [$clog2(DEPTH)-1:0] wr_addr;
reg half_wr;  // only for fp8

wire fp16 = d_dtype;
wire fp8  = ~d_dtype;

assign in_ready_o = 1'b1; // 简化处理

always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        wr_addr <= 0;
        half_wr <= 0;
    end
    else if(in_valid_i) begin
        if(fp16) begin
            wr_addr <= wr_addr + 1;
        end
        else begin
            // fp8: 写两次组成一行
            if(half_wr) begin
                wr_addr <= wr_addr + 1;
                half_wr <= 0;
            end
            else begin
                half_wr <= 1;
            end
        end
    end
end

////////////////////////////////////////////////////////////
// write enable
////////////////////////////////////////////////////////////
reg [1:0] wr_en;

always @(*) begin
    if(in_valid_i) begin
        if(fp16)
            wr_en = 2'b11;   // 整行写
        else
            wr_en = half_wr ? 2'b10 : 2'b01; // 半行写
    end
    else
        wr_en = 2'b00;
end

////////////////////////////////////////////////////////////
// write data mux
////////////////////////////////////////////////////////////
reg [IN_BUS_WIDTH-1:0] wr_data;

always @(*) begin
    if(fp16) begin
        wr_data = conv_out;
    end
    else begin
        // fp8 只有低512bit有效
        if(half_wr)
            wr_data = {conv_out[511:0], 512'd0}; // 写 upper
        else
            wr_data = {512'd0, conv_out[511:0]}; // 写 lower
    end
end

////////////////////////////////////////////////////////////
// read control
////////////////////////////////////////////////////////////
reg [$clog2(DEPTH)-1:0] rd_addr;
reg half_rd;

always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        rd_addr <= 0;
        half_rd <= 0;
    end
    else if(out_ready_i) begin

        if(half_rd) begin
            rd_addr <= rd_addr + 1;
            half_rd <= 0;
        end
        else begin
            half_rd <= 1;
   
        end
    end
end

////////////////////////////////////////////////////////////
// read enable
////////////////////////////////////////////////////////////
reg [1:0] rd_en;

always @(*) begin
    if(out_ready_i) begin
        // if(fp16)
        //     rd_en = 2'b11;
        // else
            rd_en = half_rd ? 2'b10 : 2'b01;
    end
    else
        rd_en = 2'b00;
end

////////////////////////////////////////////////////////////
// SRAM instance
////////////////////////////////////////////////////////////
wire [IN_BUS_WIDTH-1:0] rd_data;
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
    .rd_valid(rd_data_valid)
);

////////////////////////////////////////////////////////////
// output mux
////////////////////////////////////////////////////////////
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        d_out_data <= 0;
        out_valid_o <= 0;
    end
    else if(rd_data_valid) begin
        out_valid_o <= 1;

        // if(fp16) begin
        //     // fp16：一次吐512（拆两拍更合理，这里简单做低半）
        //     d_out_data <= rd_data[511:0];
        // end
        //else begin
            // fp8：只取半区
        if(half_rd)
            d_out_data <= rd_data[511:0];
        else
            d_out_data <= rd_data[1023:512];
 //       end
    end
    else begin
        out_valid_o <= 0;
    end
end

endmodule