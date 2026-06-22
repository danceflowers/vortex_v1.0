/*
 * Copyright (c) 2023-2024 C*Core Technology Co.,Ltd,Suzhou.
 * Ventus-RTL is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details. */
// Author: Gu, Zihan
// Description:张量计算单元执行部分（输出接入fifo）
`timescale 1ns/1ns
`include "define.v"

module tensor_core #(
        parameter VL         = `NUM_THREAD,
        parameter SHAPE_M      = 8,
        parameter SHAPE_K      = 8,
        parameter SHAPE_N      = 8,
        parameter EXPWIDTH   = 5,
        parameter PRECISION  = 4,
        parameter SUM_OUT_EXP = 8,
        parameter SUM_OUT_MAN = 13
    )(
        input                                     clk              ,
        input                                     rst_n            ,

        input   [`MATRIX_BUS_WIDTH-1:0]           s_axis_tdata_a            ,
        input   [`MATRIX_BUS_WIDTH-1:0]           s_axis_tdata_b            ,
        input   [`MATRIX_BUS_WIDTH-1:0]           s_axis_tdata_c            ,
        
        input   [`REGIDX_WIDTH+`REGEXT_WIDTH-1:0] ctrl_reg_idxw_i  ,
        input   [`DEPTH_WARP-1:0]                 ctrl_wid_i       ,

        input   [2:0]                             rm_i             ,

        // ==================== 控制与状态信号 ====================
        // input                                   en,
        // output                                  busy,
        // output    [7:0]                         irq,
        // input     [7:0]                         irq_en,
        input                                 mma_start_i,
        output                                mma_ready,  
        input                                 rd_d_en,
		output                                rd_d_busy,
        output                                result_ready_o,

        input     [1:0]                         a_type_i,
        input     [1:0]                         b_type_i,
        input     [1:0]                         c_type_i,  //2'b00:fp8(e5m2),2'b01:fp8(e4m3),2'b10:fp16,2'b11:fp32
        input     [1:0]                         d_type_i,  //todo:没有逐级传递到输出数据类型控制模块   // 11:FP32, 10:FP16, 01:E4M3, 00:E5M2

        // ==================== AXI Stream与Tensore Core之间的握手信号 ====================
        input                               s_axis_tvalid_a,
        output                              s_axis_tready_a,
     // input                               s_axis_tlast_a,

        input                               s_axis_tvalid_b,
        output                              s_axis_tready_b,
     // input                               s_axis_tlast_b,

        input                               s_axis_tvalid_c,
        output                              s_axis_tready_c,
     // input                               s_axis_tlast_c,

        input                               m_axis_tready_d,
        output                              m_axis_tvalid_d,
        output                              m_axis_tlast_d,
    
        // Align the result before writeback
        output  [SHAPE_M*SHAPE_N*`XLEN_FP8-1:0]           wb_wvd_rd_o      

    );
//对amem,bmem，cmem读启动信号进行对齐，a\bmem读有效后一周期返回数据。而对于cmem,fp8时读有效后两周期返回数据，fp16时3周期，fp32时5周期
reg [4:0]mma_start_d;
always @(posedge clk,negedge rst_n) begin
    if(!rst_n) begin
        mma_start_d <= 0;
    end
    else begin
        mma_start_d <= {mma_start_d[3:0],mma_start_i};
    end
    
end
reg c_rd_start;
reg a_rd_start;
reg b_rd_start;
always @(*) begin
        c_rd_start <= mma_start_i;
        if(c_type_i == 2'b00 || c_type_i == 2'b01) begin
            a_rd_start <= mma_start_d[1];
            b_rd_start <= mma_start_d[1];
        end
        else if(c_type_i == 2'b10) begin
            a_rd_start <= mma_start_d[2];
            b_rd_start <= mma_start_d[2];
        end
        else if(c_type_i == 2'b11) begin
            a_rd_start <= mma_start_d[4];
            b_rd_start <= mma_start_d[4];
        end
    end

wire cmem_out_valid;
wire abmem_out_valid;

wire mm_mul_add_in_ready;
wire mm_mul_add_out_valid;

// result输入数据
// todo: depends on specified output data type
wire   [SHAPE_M*SHAPE_N*`XLEN_FP22 -1:0] result_v_data_in     ;
wire   [`MATRIX_BUS_WIDTH-1:0] result_v_data_out    ;
wire                                                         result_v_in_valid    ;
wire                                                         result_v_in_ready    ;
wire                                                         result_v_out_valid   ;
wire                                                         result_v_out_ready   ;

//dtype_conv
wire [`MATRIX_BUS_WIDTH*`XLEN_FP9E5M3/`XLEN_FP8-1:0] fp9_a_o;
wire [`MATRIX_BUS_WIDTH*`XLEN_FP9E5M3/`XLEN_FP8-1:0] fp9_b_o;
wire [SHAPE_M*SHAPE_N*`XLEN_FP22-1:0] fp22_c_o; 
to_intype_con #(
    .BANK_NUM(8),
    .BANK_WIDTH(72),
    .DEPTH(8),
    .BUS_WIDTH(512)
)u_to_intype_con(
    .clk(clk),
    .rst_n(rst_n),

    .a_dtype(a_type_i),
    .b_dtype(b_type_i),
    .rd_start_i(a_rd_start),   //todo

    // A operand input
    .a_in_data(s_axis_tdata_a),
    .a_in_valid_i(s_axis_tvalid_a),
    .a_in_ready_o(s_axis_tready_a),

    // B operand input
    .b_in_data(s_axis_tdata_b),
    .b_in_valid_i(s_axis_tvalid_b),
    .b_in_ready_o(s_axis_tready_b),

    .out_ready_i(mm_mul_add_in_ready),   //下一级就绪

    // A operand output
    .a_out_data(fp9_a_o),

    // B operand output
    .b_out_data(fp9_b_o),

    // merged valid
    .out_valid_o(abmem_out_valid)  //转换有效
);

to_next_con #(
    .BANK_NUM(8),
    .BANK_WIDTH(72),
    .DEPTH(8),
    .BUS_WIDTH(512),
    .OUT_WIDTH(1408)
)U_to_next_con(
    .clk(clk),
    .rst_n(rst_n),

    .dtype(c_type_i),   
    .rd_start_i(c_rd_start),   //todo

    .in_data(s_axis_tdata_c),
    .in_valid_i(s_axis_tvalid_c),   
    .in_ready_o(mma_ready),   //可以读了
    .out_ready_i(mm_mul_add_in_ready),   //下一级就绪

    .out_data(fp22_c_o),  
    .out_valid_o(cmem_out_valid)  //转换有效
);
assign s_axis_tready_c = 1'b1;  //todo:
//实例化计算单元
 mm_mul_add #(
                    .VL       (VL       ),
                    .SHAPE_M    (SHAPE_M    ),
                    .SHAPE_K    (SHAPE_K    ),
                    .SHAPE_N    (SHAPE_N    ),
                    .EXPWIDTH (EXPWIDTH ),
                    .PRECISION(PRECISION),
                    .SUM_OUT_EXP(SUM_OUT_EXP),
                    .SUM_OUT_MAN(SUM_OUT_MAN)
                )
                u_mm_mul_add (
                    .clk            (clk                 ),
                    .rst_n          (rst_n               ),

                    .a_i            (fp9_a_o               ),
                    .b_i            (fp9_b_o               ),
                    .c_i            (fp22_c_o              ),
                    .rm_i           ({VL{rm_i}}          ),
    
                    // .ctrl_reg_idxw_i(ctrl_reg_idxw_i     ),
                    // .ctrl_warpid_i  (ctrl_wid_i          ),

                    .in_valid_i     (abmem_out_valid  && cmem_out_valid     ),     
                    .out_ready_i    (result_v_in_ready   ),

                    .in_ready_o     (  mm_mul_add_in_ready  ),
                    .out_valid_o    (  result_v_in_valid  ),
                    // todo: result data is the output of fp22-accumulator
                    .result_o       (result_v_data_in       ),
                    .fflags_o       (       )
                    // .ctrl_reg_idxw_o(tensor_ctrl_reg_idxw),
                    // .ctrl_warpid_o  (tensor_ctrl_warpid  )
                );


    
    //实例化输出ram
    to_outtype_con #(
    .BANK_NUM(16),
    .BANK_WIDTH(64),
    .DEPTH(8),
    .IN_BUS_WIDTH(SHAPE_M*SHAPE_N*`XLEN_FP22),
    .OUT_BUS_WIDTH(`MATRIX_BUS_WIDTH)
)u_to_outtype_con(
    .clk(clk),
    .rst_n(rst_n),

    .d_dtype(d_type_i),  //1:fp16,0:fp8  // 11:FP32, 10:FP16, 01:E4M3, 00:E5M2
    
    .d_in_data(result_v_data_in),
    .in_valid_i(result_v_in_valid),
    .in_ready_o(result_v_in_ready),

    .out_ready_i(result_v_out_ready), 
    .out_ready_o(result_ready_o),
    .rd_d_en(rd_d_en), 
	.rd_busy(rd_d_busy),
    .out_valid_o(result_v_out_valid),

    .d_out_data(result_v_data_out)
);

    assign result_v_out_ready = m_axis_tready_d;  //下一级ready,这个信号会触发读
    assign wb_wvd_rd_o = result_v_data_out;

    assign m_axis_tvalid_d = result_v_out_valid;
    assign m_axis_tlast_d  = result_v_out_valid;
endmodule
