/*
 * Functional Sparse+OpenTensorCore top.
 * This version is a runnable behavioral simulation package that preserves the
 * sparse module chain and top-level interfaces, while simplifying arithmetic
 * and buffering so it can compile and run standalone.
 */
`timescale 1ns/1ns
`include "define.v"

module tensor_core #(
    parameter VL           = `NUM_THREAD,
    parameter SHAPE_M      = 8,
    parameter SHAPE_K      = 8,
    parameter SHAPE_N      = 8,
    parameter EXPWIDTH     = 5,
    parameter PRECISION    = 4,
    parameter SUM_OUT_EXP  = 8,
    parameter SUM_OUT_MAN  = 13,
    parameter META_WIDTH   = SHAPE_M*(SHAPE_K/4)*`SPARSE_META_BITS_PER_GROUP     //8*2*8
)(
    input                                     clk,
    input                                     rst_n,
    input   [`MATRIX_BUS_WIDTH-1:0]           s_axis_tdata_a,
    input   [`MATRIX_BUS_WIDTH-1:0]           s_axis_tdata_b,
    input   [`MATRIX_BUS_WIDTH-1:0]           s_axis_tdata_c,
    input   [2:0]                             sparse_mode_sel_i,
    input                                     a_is_compressed_i,
    input                                     meta_valid_i,
    input   [META_WIDTH-1:0]                  s_axis_tmeta_a,
    input   [`REGIDX_WIDTH+`REGEXT_WIDTH-1:0] ctrl_reg_idxw_i,
    input   [`DEPTH_WARP-1:0]                 ctrl_wid_i,
    input   [2:0]                             rm_i,
    input                                     mma_start_i,
    output                                    mma_ready,
    input     [1:0]                           a_type_i,
    input     [1:0]                           b_type_i,
    input     [1:0]                           c_type_i,
    input                                     d_type_i,
    input                                     s_axis_tvalid_a,
    output                                    s_axis_tready_a,
    input                                     s_axis_tvalid_b,
    output                                    s_axis_tready_b,
    input                                     s_axis_tvalid_c,
    output                                    s_axis_tready_c,
    input                                     m_axis_tready_d,
    output                                    m_axis_tvalid_d,
    output                                    m_axis_tlast_d,
    output  [SHAPE_M*SHAPE_N*`XLEN_FP8-1:0]   wb_wvd_rd_o
);

wire launch_req;
reg  launch_d1;
wire [`MATRIX_BUS_WIDTH-1:0] a_parsed_data;
wire [META_WIDTH-1:0]        a_parsed_meta;
wire [2:0]                   sparse_mode_eff;
wire                         a_parser_valid;
wire                         a_parser_ready;
wire [SHAPE_M*SHAPE_K*`XLEN_FP9E5M3-1:0] fp9_a_payload_o;
wire [SHAPE_N*SHAPE_K*`XLEN_FP9E5M3-1:0] fp9_b_dense_o;
wire [META_WIDTH-1:0] a_meta_o;
wire [2:0] a_sparse_mode_o;
wire a_meta_valid_o;
wire abmem_out_valid;
wire [SHAPE_M*SHAPE_K*`XLEN_FP9E5M3-1:0] fp9_a_sparse_aligned;
wire [SHAPE_M*SHAPE_K-1:0] sparse_lane_valid;
wire [SHAPE_M*SHAPE_K*`XLEN_FP9E5M3-1:0] fp9_a_selected;
wire [SHAPE_N*SHAPE_K*`XLEN_FP9E5M3-1:0] fp9_b_selected;
wire [SHAPE_M*SHAPE_N*`XLEN_FP22-1:0] fp22_c_o;
wire cmem_out_valid;
wire mm_mul_add_in_ready;
wire result_v_in_valid;
wire result_v_in_ready;
wire result_v_out_valid;
wire result_v_out_ready;
wire [`MATRIX_BUS_WIDTH*2-1:0] result_v_data_in;
wire [`MATRIX_BUS_WIDTH-1:0] result_v_data_out;

assign launch_req = mma_start_i & s_axis_tvalid_a & s_axis_tvalid_b & s_axis_tvalid_c;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        launch_d1 <= 1'b0;
    else
        launch_d1 <= launch_req;
end

input_parser_bypass #(
    .BUS_WIDTH(`MATRIX_BUS_WIDTH),
    .META_WIDTH(META_WIDTH)
) u_input_parser_bypass (
    .clk(clk),
    .rst_n(rst_n),
    .sparse_mode_sel_i(sparse_mode_sel_i),
    .a_is_compressed_i(a_is_compressed_i),
    .meta_i(s_axis_tmeta_a),
    .meta_valid_i(meta_valid_i),
    .a_data_i(s_axis_tdata_a),
    .a_valid_i(s_axis_tvalid_a),
    .a_ready_o(s_axis_tready_a),
    .out_ready_i(1'b1),
    .a_data_o(a_parsed_data),
    .meta_o(a_parsed_meta),
    .sparse_mode_sel_o(sparse_mode_eff),
    .meta_valid_o(),
    .out_valid_o(a_parser_valid)
);

assign s_axis_tready_b = 1'b1;
assign s_axis_tready_c = 1'b1;
assign mma_ready = 1'b1;

to_intype_con #(
    .BANK_NUM(8),
    .BANK_WIDTH(72),
    .DEPTH(8),
    .BUS_WIDTH(512),
    .SHAPE_M(SHAPE_M),
    .SHAPE_K(SHAPE_K),
    .META_WIDTH(META_WIDTH)
) u_to_intype_con (
    .clk(clk),
    .rst_n(rst_n),
    .a_dtype(a_type_i),
    .b_dtype(b_type_i),
    .rd_start_i(launch_d1),
    .a_in_data(a_parsed_data),
    .a_in_valid_i(a_parser_valid),
    .a_in_ready_o(a_parser_ready),
    .a_sparse_mode_i(sparse_mode_eff),
    .a_meta_i(a_parsed_meta),
    .a_meta_valid_i(meta_valid_i & a_is_compressed_i & (sparse_mode_eff != `SPARSE_MODE_DENSE)),
    .b_in_data(s_axis_tdata_b),
    .b_in_valid_i(s_axis_tvalid_b),
    .b_in_ready_o(),
    .out_ready_i(mm_mul_add_in_ready),
    .a_out_data(fp9_a_payload_o),
    .a_out_meta(a_meta_o),
    .a_out_sparse_mode(a_sparse_mode_o),
    .a_out_meta_valid(a_meta_valid_o),
    .b_out_data(fp9_b_dense_o),
    .out_valid_o(abmem_out_valid)
);

spar_mul #(
    .SHAPE_M(SHAPE_M),
    .SHAPE_K(SHAPE_K),
    .ELEM_WIDTH(`XLEN_FP9E5M3),
    .META_BITS_PER_GROUP(`SPARSE_META_BITS_PER_GROUP)
) u_spar_mul (
    .sparse_mode_sel_i((a_meta_valid_o) ? a_sparse_mode_o : `SPARSE_MODE_DENSE),
    .a_comp_i(fp9_a_payload_o),
    .meta_i(a_meta_o),
    .a_align_o(fp9_a_sparse_aligned),
    .lane_valid_o(sparse_lane_valid)
);

tc_mul_dual_path #(    //todo:this module maybe useless
    .SHAPE_M(SHAPE_M),
    .SHAPE_K(SHAPE_K),
    .SHAPE_N(SHAPE_N),
    .ELEM_WIDTH(`XLEN_FP9E5M3)
) u_tc_mul_dual_path (
    .sparse_mode_sel_i((a_meta_valid_o) ? a_sparse_mode_o : `SPARSE_MODE_DENSE),
    .dense_a_i(fp9_a_payload_o),
    .sparse_a_i(fp9_a_sparse_aligned),
    .dense_b_i(fp9_b_dense_o),
    .a_o(fp9_a_selected),
    .b_o(fp9_b_selected)
);

to_next_con #(
    .BANK_NUM(8),
    .BANK_WIDTH(64),
    .DEPTH(8),
    .BUS_WIDTH(512),
    .OUT_WIDTH(1408)
) u_to_next_con (
    .clk(clk),
    .rst_n(rst_n),
    .dtype(c_type_i),
    .rd_start_i(launch_d1),
    .in_data(s_axis_tdata_c),
    .in_valid_i(s_axis_tvalid_c),
    .in_ready_o(),
    .out_ready_i(mm_mul_add_in_ready),
    .out_data(fp22_c_o),
    .out_valid_o(cmem_out_valid)
);

mm_mul_add #(
    .VL(VL),
    .SHAPE_M(SHAPE_M),
    .SHAPE_K(SHAPE_K),
    .SHAPE_N(SHAPE_N),
    .EXPWIDTH(EXPWIDTH),
    .PRECISION(PRECISION),
    .SUM_OUT_EXP(SUM_OUT_EXP),
    .SUM_OUT_MAN(SUM_OUT_MAN)
) u_mm_mul_add (
    .clk(clk),
    .rst_n(rst_n),
    .a_i(fp9_a_selected),
    .b_i(fp9_b_selected),
    .c_i(fp22_c_o),
    .rm_i({VL{rm_i}}),
    .in_valid_i(abmem_out_valid && cmem_out_valid),
    .out_ready_i(result_v_in_ready),
    .in_ready_o(mm_mul_add_in_ready),
    .out_valid_o(result_v_in_valid),
    .result_o(result_v_data_in),
    .fflags_o()
);

to_outtype_con #(
    .BANK_NUM(16),
    .BANK_WIDTH(64),
    .DEPTH(8),
    .IN_BUS_WIDTH(1024),
    .OUT_BUS_WIDTH(`MATRIX_BUS_WIDTH)
) u_to_outtype_con (
    .clk(clk),
    .rst_n(rst_n),
    .d_dtype(d_type_i),
    .d_in_data(result_v_data_in),
    .in_valid_i(result_v_in_valid),
    .in_ready_o(result_v_in_ready),
    .out_ready_i(m_axis_tready_d),
    .out_valid_o(result_v_out_valid),
    .d_out_data(result_v_data_out)
);

assign result_v_out_ready = m_axis_tready_d;
assign wb_wvd_rd_o = result_v_data_out;
assign m_axis_tvalid_d = result_v_out_valid;
assign m_axis_tlast_d  = result_v_out_valid;

endmodule

