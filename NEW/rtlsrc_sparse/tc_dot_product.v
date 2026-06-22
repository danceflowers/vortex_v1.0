`timescale 1ns/1ns
`include "define.v"
//2026/0302
//This version extends the FP9(E5M3)multiplication results to FP10(E6M3) before accumulation
module tc_dot_product #(
    parameter SHAPE_K = 8,
    parameter EXPWIDTH = 5,
    parameter PRECISION = 4,
    parameter SUM_OUT_EXP = 8,
    parameter SUM_OUT_MAN = 13
)(
    input clk,
    input rst_n,
    input [SHAPE_K*`XLEN_FP9E5M3-1:0] a_i,
    input [SHAPE_K*`XLEN_FP9E5M3-1:0] b_i,
    input [`XLEN_FP22-1:0] c_i,

    input [2:0] rm_i,

    input in_valid_i,
    input out_ready_i,

    output in_ready_o,
    output out_valid_o,
    output [`XLEN_FP16-1:0] result_o,
    output [4:0] fflags_o
);
    localparam EXTEND_EXPWIDTH = EXPWIDTH + 1;
    localparam EXTEND_PRECISION = PRECISION*2 ;

    wire [2:0] mctrl_rm;
    wire [`XLEN_FP22-1:0] mctrl_c;

    wire muls_in_ready [0:SHAPE_K-1];
    wire muls_out_valid [0:SHAPE_K-1];
    wire muls_out_ready [0:SHAPE_K-1];

    wire [EXTEND_EXPWIDTH + EXTEND_PRECISION-1:0] muls_result [0:SHAPE_K-1];

    wire [4:0] muls_fflags [0:SHAPE_K-1];
    wire [`XLEN_FP22-1:0] muls_ctrl_c [0:SHAPE_K-1];
    wire [2:0] muls_ctrl_rm [0:SHAPE_K-1];
    
    wire [2:0] actrls_rm [0:2];  // 3个就够了
    wire adds_in_ready [0:2];
    wire adds_out_valid [0:2];
    wire adds_out_ready [0:2];

    wire [SUM_OUT_EXP+SUM_OUT_MAN:0] adds_result [0:2];

    wire [4:0] adds_fflags [0:2];
    wire [`XLEN_FP22-1:0] adds_ctrl_c [0:2];
    wire [2:0] adds_ctrl_rm [0:2];


    
    wire finaladd_in_ready;
    wire finaladd_out_valid;
    wire [`XLEN_FP22-1:0] finaladd_result;
    wire [4:0] finaladd_fflags;
    wire [`XLEN_FP22-1:0] finaladd_ctrl_c;
    wire [2:0] finaladd_ctrl_rm;
    

    wire [`XLEN_FP8-1:0] result_fp8;
    wire fp8_out_valid; 
    wire sram_in_valid;
    wire sram_in_ready;
    wire sram_out_valid;
    wire sram_out_ready;
    localparam SRAM_WIDTH = `XLEN_FP8+`XLEN_FP16 + 16 + `DEPTH_WARP;  //todo
    wire [SRAM_WIDTH-1:0] sram_data_in;
    wire [3:0] sram_in_addr;
    wire sram_wr_en;
    wire [SRAM_WIDTH-1:0] sram_data_out;
    reg out_valid_reg_d1, out_valid_reg_d2;

    assign actrls_rm[0] = muls_ctrl_rm[0];
    assign mctrl_rm = rm_i;
    assign mctrl_c = c_i;

    genvar i;
    generate
        for(i=0;i<SHAPE_K;i=i+1) begin:muls
            tc_mul_pipe #(.EXPWIDTH (EXPWIDTH ), .PRECISION(PRECISION), .LATENCY(2))
                        U_tc_mul_pipe (
                            .clk(clk), 
                            .rst_n(rst_n),

                            .a_i(a_i[(i+1)*`XLEN_FP9E5M3-1:i*`XLEN_FP9E5M3]),
                            .b_i(b_i[(i+1)*`XLEN_FP9E5M3-1:i*`XLEN_FP9E5M3]),

                            .rm_i(mctrl_rm), 
                            .ctrl_c_i(mctrl_c), 
                            .ctrl_rm_i(mctrl_rm),

                            .in_valid_i(in_valid_i), 
                            .out_ready_i(muls_out_ready[i]),
                            .in_ready_o(muls_in_ready[i]), 
                            .out_valid_o(muls_out_valid[i]),

                            .result_o(), 
                            .result_extend_o(muls_result[i]),
                            .fflags_o(muls_fflags[i]),
                            .ctrl_c_o(muls_ctrl_c[i]), 
                            .ctrl_rm_o(muls_ctrl_rm[i])
                        );
        end
    endgenerate

    genvar k;
    generate
    for(k=0;k<2;k=k+1) begin:add4
        tc_add4_pipe #(
            .LATENCY(4),
            .IN_EXP_WIDTH(EXTEND_EXPWIDTH),
            .IN_MAN_WIDTH(EXTEND_PRECISION - 1),
            .OUT_EXP_WIDTH(SUM_OUT_EXP),
            .OUT_MAN_WIDTH(SUM_OUT_MAN)
        ) U_tc_add4_pipe (
            .clk(clk), .rst_n(rst_n),
            .in_i({muls_result[k*4+0],muls_result[k*4+1],muls_result[k*4+2],muls_result[k*4+3]}),
            .rm_i(actrls_rm[k]),
            .ctrl_c_i(muls_ctrl_c[k]),
            .ctrl_rm_i(muls_ctrl_rm[k]),
            //.ctrl_reg_idxw_i(muls_ctrl_reg_idxw[k]),
            //.ctrl_warpid_i(muls_ctrl_warpid[k]),

            .in_valid_i(muls_out_valid[k]),
            .in_ready_o(adds_in_ready[k]),
            .out_ready_i(adds_out_ready[k]),
            .out_valid_o(adds_out_valid[k]),
            .result_o(adds_result[k]),

            .fflags_o(adds_fflags[k]),
            .ctrl_c_o(adds_ctrl_c[k]), 
            .ctrl_rm_o(adds_ctrl_rm[k])
            //.ctrl_reg_idxw_o(adds_ctrl_reg_idxw[k]), 
            //.ctrl_warpid_o(adds_ctrl_warpid[k])
        );
        assign muls_out_ready[k*4+0] = adds_in_ready[k];
        assign muls_out_ready[k*4+1] = adds_in_ready[k];
        assign muls_out_ready[k*4+2] = adds_in_ready[k];
        assign muls_out_ready[k*4+3] = adds_in_ready[k];
    end
    endgenerate

 tc_add_pipe #(.EXPWIDTH (SUM_OUT_EXP), .PRECISION(SUM_OUT_MAN+1), .LATENCY(2))
            U_tc_add_pipe (
                .clk(clk), 
                .rst_n(rst_n),
                .a_i(adds_result[1]),
                .b_i(adds_result[0]),
                .rm_i(adds_ctrl_rm[0]), 
                .ctrl_c_i(adds_ctrl_c[0]),
                .ctrl_rm_i(adds_ctrl_rm[0]),
                // .ctrl_reg_idxw_i(adds_ctrl_reg_idxw[0]),
                // .ctrl_warpid_i(adds_ctrl_warpid[0]),

                .in_valid_i(adds_out_valid[0]),
                .out_ready_i(adds_out_ready[2]),
                .in_ready_o(adds_in_ready[2]),
                .out_valid_o(adds_out_valid[2]),

                .result_o(adds_result[2]),
                .fflags_o(adds_fflags[2]),

                .ctrl_c_o(adds_ctrl_c[2]),
                .ctrl_rm_o(adds_ctrl_rm[2])
                // .ctrl_reg_idxw_o(adds_ctrl_reg_idxw[2]),
                // .ctrl_warpid_o(adds_ctrl_warpid[2  ])
            );

    assign adds_out_ready[0] = adds_in_ready[2];  //后一级ready
    assign adds_out_ready[1] = adds_in_ready[2];
    assign adds_out_ready[2] = finaladd_in_ready;


    wire [EXTEND_EXPWIDTH+PRECISION-1:0] op_a_raw = adds_result[2]; 
    wire [`XLEN_FP22-1:0]                op_b_raw = adds_ctrl_c[2];

    wire [`XLEN_FP22-1:0] op_a_fp22 = adds_result[2]; 
    wire [`XLEN_FP22-1:0] op_b_fp22 = adds_ctrl_c[2];



    tc_add_pipe #(.EXPWIDTH (8), .PRECISION(14), .LATENCY(2))
                U_final_add (
                    .clk            (clk                        ),
                    .rst_n          (rst_n                      ),
                    .a_i            (op_a_fp22),
                    .b_i            (op_b_fp22),
                    .rm_i           (adds_ctrl_rm[2]    ),
                    .ctrl_c_i       (adds_ctrl_c[2]),  //todo:无关紧要
                    .ctrl_rm_i      (adds_ctrl_rm[2]    ),
                    // .ctrl_reg_idxw_i(adds_ctrl_reg_idxw[2]),
                    // .ctrl_warpid_i  (adds_ctrl_warpid[2]),
                    .in_valid_i     (adds_out_valid[2]  ),
                    .out_ready_i    ( out_ready_i              ),
                    .in_ready_o     (finaladd_in_ready          ),
                    .out_valid_o    (finaladd_out_valid         ),
                    .result_o       (finaladd_result            ),
                    .fflags_o       (finaladd_fflags            ),
                    .ctrl_c_o       (finaladd_ctrl_c            ),
                    .ctrl_rm_o      (finaladd_ctrl_rm           )
                    // .ctrl_reg_idxw_o(finaladd_ctrl_reg_idxw     ),
                    // .ctrl_warpid_o  (finaladd_ctrl_warpid       )
                );


    wire [7:0]  exp22_out_raw = finaladd_result[20:13];
    wire [12:0] mant22_out_raw = finaladd_result[12:0];
    wire [3:0]  exp8_out_new;
    wire [4:0]  exp16_out_new;
    wire [15:0] result_fp16;
    assign exp8_out_new = (exp22_out_raw < 8'd120) ? 4'd0 : (exp22_out_raw[3:0] - 4'd8);
    assign exp16_out_new = (exp22_out_raw < 8'd112) ? 5'd0 :(exp22_out_raw[4:0] - 5'd16);
    assign result_fp8 = { finaladd_result[21], exp8_out_new, mant22_out_raw[12:10] };
    assign result_fp16 = { finaladd_result[21], exp16_out_new, mant22_out_raw[12:3] };


    // fp22_to_fp8_con U_fp22_to_fp8_con      //maybe useless
    //             (
    //                 .clk(clk), .rst_n(rst_n), .out_type_i(type_ab_sub), .data_i(finaladd_result),
    //                 .data_o(), .in_valid_i(finaladd_out_valid), .out_ready_i(sram_in_ready),
    //                 .in_ready_o(), .out_valid_o(fp8_out_valid) 
    //             );


    // singleportSRAM #(.D_WIDTH(`XLEN_FP8+`XLEN_FP16+16+`DEPTH_WARP), .A_WIDTH(4))
    //             U_sram (
    //                 .clk(clk), .rst_n(rst_n), .w_en_i(sram_wr_en), .addr(4'd0),
    //                 .w_valid_i(sram_in_valid), .w_data_i(sram_data_in), .r_ready_i(sram_out_ready),
    //                 .w_ready_o(sram_in_ready), .r_data_o(sram_data_out), .r_valid_o(sram_out_valid)
    //             );

    // assign sram_data_in   = {result_fp8, finaladd_fflags, finaladd_ctrl_c, finaladd_ctrl_rm, finaladd_ctrl_reg_idxw, finaladd_ctrl_warpid};
    // assign sram_in_valid  = finaladd_out_valid; 
    // assign sram_wr_en     = sram_in_valid;
    // assign sram_out_ready = out_ready_i;
    // assign sram_in_addr   = 4'd0;
    
 
    // assign result_o       = sram_data_out[42:35]; 
    // assign fflags_o       = sram_data_out[34:30];
    // assign ctrl_reg_idxw_o= sram_data_out[8+`DEPTH_WARP-1-:8];
    // assign ctrl_warpid_o  = sram_data_out[`DEPTH_WARP-1:0];


    // always @(posedge clk or negedge rst_n) begin
    //     if (!rst_n) begin
    //         out_valid_reg_d1 <= 1'b0;
    //         out_valid_reg_d2 <= 1'b0;
    //     end else begin
    //         out_valid_reg_d1 <= sram_in_valid; 
    //         out_valid_reg_d2 <= out_valid_reg_d1; 
    //     end
    // end
    // assign out_valid_o = out_valid_reg_d2;
    // assign in_ready_o     = muls_in_ready[SHAPE_K-1];

    assign result_o       = result_fp16; 
    assign fflags_o       = finaladd_fflags;
    assign out_valid_o    = finaladd_out_valid;
    assign in_ready_o     = muls_in_ready[SHAPE_K-1];



endmodule
