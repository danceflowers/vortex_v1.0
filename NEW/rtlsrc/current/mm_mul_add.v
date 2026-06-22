
`timescale 1ns/1ns
`include "define.v"

module mm_mul_add #(
        parameter VL        = `NUM_THREAD,
        parameter SHAPE_M     = 8,
        parameter SHAPE_K     = 8,
        parameter SHAPE_N     = 8,
        parameter EXPWIDTH  = 5,
        parameter PRECISION = 4,
        parameter SUM_OUT_EXP = 8,
        parameter SUM_OUT_MAN = 13
    )(
        input                                   clk            ,
        input                                   rst_n          ,

        input   [SHAPE_M*SHAPE_K*`XLEN_FP9E5M3-1:0]   a_i            ,
        input   [SHAPE_N*SHAPE_K*`XLEN_FP9E5M3-1:0]   b_i            ,
        input   [SHAPE_M*SHAPE_N*`XLEN_FP22-1:0]   c_i            ,

        input   [VL*3-1:0]                      rm_i   ,
        // input [`REGIDX_WIDTH + `REGEXT_WIDTH - 1 : 0] ctrl_reg_idxw_i, 
        // input [`DEPTH_WARP - 1 : 0]            ctrl_warpid_i,
        input                                   in_valid_i     ,
        input                                   out_ready_i    ,

        output                                  in_ready_o     ,
        output                                  out_valid_o    ,
        
        output  [SHAPE_M*SHAPE_N*`XLEN_FP22-1:0]   result_o ,  
        output [(SHAPE_M * SHAPE_N * 5) - 1 : 0] fflags_o
		// output  [7:0]                        ctrl_reg_idxw_o,
        // output  [`DEPTH_WARP-1:0]                ctrl_warpid_o
    );

    wire                                    tc_array_in_ready     [0:(SHAPE_M * SHAPE_N)-1];
    wire                                    tc_array_out_valid    [0:(SHAPE_M * SHAPE_N)-1];
    // wire    [7:0]                           tc_array_ctrl_reg_idxw[0:(SHAPE_M * SHAPE_N)-1];
    // wire    [`DEPTH_WARP-1:0]               tc_array_ctrl_warpid  [0:(SHAPE_M * SHAPE_N)-1];

    //例化SHAPE_M*SHAPE_N次tc_dot_product
    genvar i,j;
    generate
        for(i=0;i<SHAPE_M;i=i+1) begin : A1
            for(j=0;j<SHAPE_N;j=j+1) begin : A2
                tc_dot_product #(
                                   .SHAPE_K  (SHAPE_K    ),
                                   .EXPWIDTH (EXPWIDTH ),
                                   .PRECISION(PRECISION),
                                   .SUM_OUT_EXP(SUM_OUT_EXP),
                                   .SUM_OUT_MAN(SUM_OUT_MAN)
                               )
                               U_tc_dot_array (
                                   .clk            (clk                                                                            ),
                                   .rst_n          (rst_n                                                                          ),
                                   // New configuration method for bitwidth
                                   .a_i            (a_i[(i+1)*SHAPE_K*`XLEN_FP9E5M3-1:i*SHAPE_K*`XLEN_FP9E5M3]           ),
                                   .b_i            (b_i[(j+1)*SHAPE_K*`XLEN_FP9E5M3-1:j*SHAPE_K*`XLEN_FP9E5M3]           ),
                                   .c_i ( c_i[(i*SHAPE_N+j+1)*`XLEN_FP22-1:(i*SHAPE_N+j)*`XLEN_FP22] ),

                                   .rm_i           (rm_i[2:0]                                                                      ),
                                //    .ctrl_reg_idxw_i(ctrl_reg_idxw_i                                                                ),
                                //    .ctrl_warpid_i  (ctrl_warpid_i                                                                  ),


                                    // add c handshake
                                   .in_valid_i     (in_valid_i                                                                     ),
                                   .out_ready_i    (out_ready_i                                                                    ),
                                   .in_ready_o     (tc_array_in_ready[i*SHAPE_N+j]                                                   ),
                                   .out_valid_o    (tc_array_out_valid[i*SHAPE_N+j]                                                  ),

                                   .result_o       (result_o[(i*SHAPE_N+j+1)*`XLEN_FP22-1:(i*SHAPE_N+j)*`XLEN_FP22]),
                                   .fflags_o       (fflags_o[(i*SHAPE_N+j+1)*5-1:(i*SHAPE_N+j)*5]                                      )
                                //    .ctrl_reg_idxw_o(tc_array_ctrl_reg_idxw[i*SHAPE_N+j]                                              ),
                                //    .ctrl_warpid_o  (tc_array_ctrl_warpid[i*SHAPE_N+j]                                                )
                               );

            end
        end
    

    // assign ctrl_reg_idxw_o = tc_array_ctrl_reg_idxw[0];
    // assign ctrl_warpid_o   = tc_array_ctrl_warpid[0];

    assign in_ready_o  = tc_array_in_ready[0];
    assign out_valid_o = tc_array_out_valid[0];


            
        
    endgenerate


endmodule
