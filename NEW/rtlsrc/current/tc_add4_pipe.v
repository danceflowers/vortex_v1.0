`timescale 1ns / 1ns
`include "define.v"

module tc_add4_pipe #(
    parameter LATENCY = 4,
    parameter IN_EXP_WIDTH = 6,
    parameter IN_MAN_WIDTH = 3,
    parameter OUT_EXP_WIDTH = 8,
    parameter OUT_MAN_WIDTH = 13
)(
    input                               clk,
    input                               rst_n,

    input  [(IN_EXP_WIDTH+IN_MAN_WIDTH+1)*4-1:0] in_i,
    input  [2:0]                         rm_i,

    input   [`XLEN_FP22-1:0]             ctrl_c_i,
    input   [2:0]                        ctrl_rm_i,
    // input   [7:0]                        ctrl_reg_idxw_i,
    // input   [`DEPTH_WARP-1:0]            ctrl_warpid_i,

    input                                in_valid_i,
    input                                out_ready_i,

    output                               in_ready_o,
    output                               out_valid_o,

    output  [4:0]                        fflags_o,
    output  [OUT_EXP_WIDTH+OUT_MAN_WIDTH:0] result_o,

    output  [`XLEN_FP22-1:0]             ctrl_c_o,
    output  [2:0]                        ctrl_rm_o
    // output  [7:0]                        ctrl_reg_idxw_o,
    // output  [`DEPTH_WARP-1:0]            ctrl_warpid_o
);


///////////////////////////////////////////////////////////////
//// valid pipeline
///////////////////////////////////////////////////////////////

reg [LATENCY-1:0] valid_pipe;

wire stall;
assign stall = valid_pipe[LATENCY-1] & ~out_ready_i;

assign in_ready_o = ~stall;

always @(posedge clk or negedge rst_n) begin
    if(!rst_n)
        valid_pipe <= 0;
    else if(!stall)
        valid_pipe <= {valid_pipe[LATENCY-2:0], in_valid_i};
end

assign out_valid_o = valid_pipe[LATENCY-1];

///////////////////////////////////////////////////////////////
//// ctrl pipeline
///////////////////////////////////////////////////////////////

reg [`XLEN_FP22-1:0]   ctrl_c_pipe      [LATENCY-1:0];
reg [2:0]              ctrl_rm_pipe     [LATENCY-1:0];
// reg [7:0]              ctrl_reg_pipe    [LATENCY-1:0];
// reg [`DEPTH_WARP-1:0]  ctrl_warp_pipe   [LATENCY-1:0];

integer i;

always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        for(i=0;i<LATENCY;i=i+1) begin
            ctrl_c_pipe[i]    <= 0;
            ctrl_rm_pipe[i]   <= 0;
            // ctrl_reg_pipe[i]  <= 0;
            // ctrl_warp_pipe[i] <= 0;
        end
    end
    else if(!stall) begin

        ctrl_c_pipe[0]    <= ctrl_c_i;
        ctrl_rm_pipe[0]   <= ctrl_rm_i;
        // ctrl_reg_pipe[0]  <= ctrl_reg_idxw_i;
        // ctrl_warp_pipe[0] <= ctrl_warpid_i;

        for(i=1;i<LATENCY;i=i+1) begin
            ctrl_c_pipe[i]    <= ctrl_c_pipe[i-1];
            ctrl_rm_pipe[i]   <= ctrl_rm_pipe[i-1];
            // ctrl_reg_pipe[i]  <= ctrl_reg_pipe[i-1];
            // ctrl_warp_pipe[i] <= ctrl_warp_pipe[i-1];
        end
    end
end

assign ctrl_c_o        = ctrl_c_pipe[LATENCY-1];
assign ctrl_rm_o       = ctrl_rm_pipe[LATENCY-1];
// assign ctrl_reg_idxw_o = ctrl_reg_pipe[LATENCY-1];
// assign ctrl_warpid_o   = ctrl_warp_pipe[LATENCY-1];

///////////////////////////////////////////////////////////////
//// fpadd4 core instance
///////////////////////////////////////////////////////////////


fpadd4 #(
    .IN_EXP_WIDTH (IN_EXP_WIDTH),
    .IN_MAN_WIDTH (IN_MAN_WIDTH),
    .OUT_EXP_WIDTH (OUT_EXP_WIDTH),
    .OUT_MAN_WIDTH (OUT_MAN_WIDTH),
    .LATENCY (LATENCY)
) u_fpadd4 (
    .clk        (clk),
    .rst_n      (rst_n),

    .in_i       (in_i),
    .rm_i       (rm_i),

    .result_o   (result_o),
    .fflags_o   (fflags_o)
);


endmodule