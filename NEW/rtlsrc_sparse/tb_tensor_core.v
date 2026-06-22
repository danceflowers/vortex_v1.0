`timescale 1ns/1ns
`include "define.v"

module tb_tensor_core;

// =====================================================================
// Parameters
// =====================================================================
parameter SHAPE_M    = 8;
parameter SHAPE_K    = 8;
parameter SHAPE_N    = 8;
parameter VL         = `NUM_THREAD;
parameter EXPWIDTH   = 5;
parameter PRECISION  = 4;
parameter CLK_PERIOD = 10;
localparam MATRIX_BUS = `MATRIX_BUS_WIDTH;
localparam META_WIDTH = SHAPE_M*(SHAPE_K/4)*`SPARSE_META_BITS_PER_GROUP;

// a/b type encoding
localparam ab_FP8E5M2 = 2'b00;
localparam ab_FP8E4M3 = 2'b01;
localparam ab_FP16    = 2'b10;

// c type encoding
localparam c_FP8E5M2 = 2'b00;
localparam c_FP8E4M3 = 2'b01;
localparam c_FP16    = 2'b10;
localparam c_FP32    = 2'b11;

// d type encoding
localparam d_fp8  = 1'b0;
localparam d_fp16 = 1'b1;

// =====================================================================
// Interface Signals
// =====================================================================
reg clk, rst_n;
reg [MATRIX_BUS-1:0] s_axis_tdata_a, s_axis_tdata_b, s_axis_tdata_c;
reg [META_WIDTH-1:0] s_axis_tmeta_a;
reg s_axis_tvalid_a, s_axis_tvalid_b, s_axis_tvalid_c;
wire s_axis_tready_a, s_axis_tready_b, s_axis_tready_c;
reg m_axis_tready_d;
wire m_axis_tvalid_d, m_axis_tlast_d;
reg [`REGIDX_WIDTH+`REGEXT_WIDTH-1:0] ctrl_reg_idxw_i;
reg [`DEPTH_WARP-1:0] ctrl_wid_i;
reg [2:0] rm_i;
reg mma_start;
reg [1:0] a_type, b_type, c_type;
reg d_type;
reg [2:0] sparse_mode_sel_i;
reg a_is_compressed_i;
reg meta_valid_i;
wire [SHAPE_M*SHAPE_N*`XLEN_FP8-1:0] wb_wvd_rd_o;
wire mma_ready;

integer total_pass;
integer total_fail;
integer total_tests;
integer timeout_counter;

// =====================================================================
// DUT
// =====================================================================
tensor_core #(
    .VL         (VL),
    .SHAPE_M    (SHAPE_M),
    .SHAPE_K    (SHAPE_K),
    .SHAPE_N    (SHAPE_N),
    .EXPWIDTH   (EXPWIDTH),
    .PRECISION  (PRECISION),
    .SUM_OUT_EXP(8),
    .SUM_OUT_MAN(13)
) dut (
    .clk               (clk),
    .rst_n             (rst_n),
    .s_axis_tdata_a    (s_axis_tdata_a),
    .s_axis_tdata_b    (s_axis_tdata_b),
    .s_axis_tdata_c    (s_axis_tdata_c),
    .sparse_mode_sel_i (sparse_mode_sel_i),
    .a_is_compressed_i (a_is_compressed_i),
    .meta_valid_i      (meta_valid_i),
    .s_axis_tmeta_a    (s_axis_tmeta_a),
    .ctrl_reg_idxw_i   (ctrl_reg_idxw_i),
    .ctrl_wid_i        (ctrl_wid_i),
    .rm_i              (rm_i),
    .mma_start_i       (mma_start),
    .mma_ready         (mma_ready),
    .a_type_i          (a_type),
    .b_type_i          (b_type),
    .c_type_i          (c_type),
    .d_type_i          (d_type),
    .s_axis_tvalid_a   (s_axis_tvalid_a),
    .s_axis_tready_a   (s_axis_tready_a),
    .s_axis_tvalid_b   (s_axis_tvalid_b),
    .s_axis_tready_b   (s_axis_tready_b),
    .s_axis_tvalid_c   (s_axis_tvalid_c),
    .s_axis_tready_c   (s_axis_tready_c),
    .m_axis_tready_d   (m_axis_tready_d),
    .m_axis_tvalid_d   (m_axis_tvalid_d),
    .m_axis_tlast_d    (m_axis_tlast_d),
    .wb_wvd_rd_o       (wb_wvd_rd_o)
);

// =====================================================================
// Clock & Reset
// =====================================================================
initial begin clk = 0; forever #(CLK_PERIOD/2) clk = ~clk; end
initial begin rst_n = 0; #(CLK_PERIOD*20); rst_n = 1; end

// =====================================================================
// Utility Functions: emulate current behavioral datapath exactly
// =====================================================================
function signed [31:0] sx8;
    input [7:0] x;
    begin
        sx8 = {{24{x[7]}}, x};
    end
endfunction

function signed [31:0] sx9;
    input [8:0] x;
    begin
        sx9 = {{23{x[8]}}, x};
    end
endfunction

function signed [31:0] sx16;
    input [15:0] x;
    begin
        sx16 = {{16{x[15]}}, x};
    end
endfunction

function signed [31:0] sx22;
    input [21:0] x;
    begin
        sx22 = {{10{x[21]}}, x};
    end
endfunction

function signed [31:0] conv_ab_lane_from_bus;
    input [1:0] dtype;
    input [MATRIX_BUS-1:0] bus;
    input integer idx;
    reg [8:0] fp9_lane;
    begin
        fp9_lane = 9'd0;
        if (dtype == ab_FP16) begin
            if (idx < 32)
                fp9_lane = {bus[idx*16+15], bus[idx*16+14 -: 8]};
            else
                fp9_lane = 9'd0;
        end else begin
            fp9_lane = {bus[idx*8+7], bus[idx*8 +: 8]};
        end
        conv_ab_lane_from_bus = sx9(fp9_lane);
    end
endfunction

function signed [31:0] conv_c_lane_from_bus;
    input [1:0] dtype;
    input [MATRIX_BUS-1:0] bus;
    input integer idx;
    reg [21:0] fp22_lane;
    begin
        fp22_lane = 22'd0;
        case (dtype)
            c_FP16: begin
                if (idx < 32)
                    fp22_lane = {{6{bus[idx*16+15]}}, bus[idx*16 +: 16]};
            end
            c_FP32: begin
                if (idx < 16)
                    fp22_lane = bus[idx*32 +: 22];
            end
            default: begin
                fp22_lane = {{14{bus[idx*8+7]}}, bus[idx*8 +: 8]};
            end
        endcase
        conv_c_lane_from_bus = sx22(fp22_lane);
    end
endfunction

function signed [31:0] sparse_a_lane_from_bus_fp8;
    input [MATRIX_BUS-1:0] bus;
    input [META_WIDTH-1:0] meta;
    input [2:0] mode;
    input integer idx;
    integer r, c, g;
    integer meta_base;
    integer payload_base;
    integer row_base;
    reg [1:0] idx0;
    reg [1:0] idx1;
    reg found;
    reg [8:0] fp9_lane;
    begin
        found = 1'b0;
        fp9_lane = 9'd0;
        r = idx / 8;
        c = idx % 8;
        g = c / 4;
        row_base = r * 8;
        meta_base = (r * (SHAPE_K/4) + g) * `SPARSE_META_BITS_PER_GROUP;
        idx0 = meta[meta_base +: 2];
        idx1 = meta[meta_base+2 +: 2];
        if (mode == `SPARSE_MODE_2TO4) begin
            if (c == g*4 + idx0) begin
                payload_base = row_base + g*2;
                fp9_lane = {bus[payload_base*8+7], bus[payload_base*8 +: 8]};
                found = 1'b1;
            end else if (c == g*4 + idx1) begin
                payload_base = row_base + g*2 + 1;
                fp9_lane = {bus[payload_base*8+7], bus[payload_base*8 +: 8]};
                found = 1'b1;
            end
        end else if (mode == `SPARSE_MODE_1TO4) begin
            if (c == g*4 + idx0) begin
                payload_base = row_base + g;
                fp9_lane = {bus[payload_base*8+7], bus[payload_base*8 +: 8]};
                found = 1'b1;
            end
        end else begin
            fp9_lane = {bus[idx*8+7], bus[idx*8 +: 8]};
            found = 1'b1;
        end

        if (found)
            sparse_a_lane_from_bus_fp8 = sx9(fp9_lane);
        else
            sparse_a_lane_from_bus_fp8 = 32'sd0;
    end
endfunction

// =====================================================================
// System Init
// =====================================================================
task sys_init;
begin
    s_axis_tdata_a  = 0;
    s_axis_tdata_b  = 0;
    s_axis_tdata_c  = 0;
    s_axis_tmeta_a  = 0;
    s_axis_tvalid_a = 0;
    s_axis_tvalid_b = 0;
    s_axis_tvalid_c = 0;
    m_axis_tready_d = 1;
    ctrl_reg_idxw_i = 0;
    ctrl_wid_i      = 0;
    rm_i            = 0;
    mma_start       = 0;
    a_type          = ab_FP8E5M2;
    b_type          = ab_FP8E5M2;
    c_type          = c_FP8E5M2;
    d_type          = d_fp8;
    sparse_mode_sel_i = `SPARSE_MODE_DENSE;
    a_is_compressed_i = 1'b0;
    meta_valid_i      = 1'b0;
    total_pass = 0;
    total_fail = 0;
    total_tests = 0;
end
endtask

// =====================================================================
// Packing helpers
// =====================================================================
task build_uniform_ab_bus;
    input [1:0] dtype;
    input [15:0] raw_val;
    output reg [MATRIX_BUS-1:0] bus;
    integer k;
    begin
        bus = {MATRIX_BUS{1'b0}};
        if (dtype == ab_FP16) begin
            for (k = 0; k < 32; k = k + 1)
                bus[k*16 +: 16] = raw_val;
        end else begin
            for (k = 0; k < 64; k = k + 1)
                bus[k*8 +: 8] = raw_val[7:0];
        end
    end
endtask

task build_uniform_c_bus;
    input [1:0] dtype;
    input [31:0] raw_val;
    output reg [MATRIX_BUS-1:0] bus;
    integer k;
    begin
        bus = {MATRIX_BUS{1'b0}};
        case (dtype)
            c_FP16: begin
                for (k = 0; k < 32; k = k + 1)
                    bus[k*16 +: 16] = raw_val[15:0];
            end
            c_FP32: begin
                for (k = 0; k < 16; k = k + 1)
                    bus[k*32 +: 32] = raw_val;
            end
            default: begin
                for (k = 0; k < 64; k = k + 1)
                    bus[k*8 +: 8] = raw_val[7:0];
            end
        endcase
    end
endtask

task pack_matrix8;
    input reg signed [7:0] arr [0:63];
    output reg [MATRIX_BUS-1:0] packed_bus;
    integer idx;
    begin
        packed_bus = {MATRIX_BUS{1'b0}};
        for (idx = 0; idx < 64; idx = idx + 1)
            packed_bus[idx*8 +: 8] = arr[idx][7:0];
    end
endtask

task pack_meta;
    input reg [7:0] marr [0:15];
    output reg [META_WIDTH-1:0] packed_meta;
    integer idx;
    begin
        packed_meta = {META_WIDTH{1'b0}};
        for (idx = 0; idx < 16; idx = idx + 1)
            packed_meta[idx*8 +: 8] = marr[idx];
    end
endtask


// =====================================================================
// Asymmetric / non-uniform packing helpers
// =====================================================================
task build_asym_ab_bus;
    input [1:0] dtype;
    input integer pattern_id;
    output reg [MATRIX_BUS-1:0] bus;
    integer r, c, idx;
    reg signed [15:0] v16;
    reg signed [7:0] v8;
    begin
        bus = {MATRIX_BUS{1'b0}};
        if (dtype == ab_FP16) begin
            for (idx = 0; idx < 32; idx = idx + 1) begin
                r = idx / 8;
                c = idx % 8;
                case (pattern_id)
                    0: v16 = (r*17 - c*9 + 5);
                    1: v16 = ((r[0] ? -1 : 1) * (r*11 + c*7 + 3));
                    default: v16 = (r*13 + c*5 - 21);
                endcase
                bus[idx*16 +: 16] = v16[15:0];
            end
        end else begin
            for (idx = 0; idx < 64; idx = idx + 1) begin
                r = idx / 8;
                c = idx % 8;
                case (pattern_id)
                    0: v8 = (r*5 - c*3 + 7);
                    1: v8 = ((c[0] ? -1 : 1) * (r*4 + c + 2));
                    default: v8 = (r*3 + c*2 - 11);
                endcase
                bus[idx*8 +: 8] = v8[7:0];
            end
        end
    end
endtask

task build_asym_c_bus;
    input [1:0] dtype;
    input integer pattern_id;
    output reg [MATRIX_BUS-1:0] bus;
    integer r, c, idx;
    reg signed [31:0] v32;
    reg signed [15:0] v16;
    reg signed [7:0] v8;
    begin
        bus = {MATRIX_BUS{1'b0}};
        case (dtype)
            c_FP32: begin
                for (idx = 0; idx < 16; idx = idx + 1) begin
                    r = idx / 8;
                    c = idx % 8;
                    case (pattern_id)
                        0: v32 = (r*101 - c*17 + 9);
                        1: v32 = ((idx[0] ? -1 : 1) * (r*55 + c*19 + 7));
                        default: v32 = (r*37 + c*23 - 41);
                    endcase
                    bus[idx*32 +: 32] = v32[31:0];
                end
            end
            c_FP16: begin
                for (idx = 0; idx < 32; idx = idx + 1) begin
                    r = idx / 8;
                    c = idx % 8;
                    case (pattern_id)
                        0: v16 = (r*23 - c*6 + 4);
                        1: v16 = ((r[0] ? -1 : 1) * (r*9 + c*3 + 2));
                        default: v16 = (r*15 + c*7 - 13);
                    endcase
                    bus[idx*16 +: 16] = v16[15:0];
                end
            end
            default: begin
                for (idx = 0; idx < 64; idx = idx + 1) begin
                    r = idx / 8;
                    c = idx % 8;
                    case (pattern_id)
                        0: v8 = (r*2 - c + 3);
                        1: v8 = ((idx[0] ? -1 : 1) * (r + c + 1));
                        default: v8 = (r*4 + c*3 - 9);
                    endcase
                    bus[idx*8 +: 8] = v8[7:0];
                end
            end
        endcase
    end
endtask
// =====================================================================
// Reference models
// =====================================================================
task calc_dense_ref_from_buses;
    input [MATRIX_BUS-1:0] A_bus;
    input [1:0] A_dtype;
    input [MATRIX_BUS-1:0] B_bus;
    input [1:0] B_dtype;
    input [MATRIX_BUS-1:0] C_bus;
    input [1:0] C_dtype;
    output reg signed [7:0] REF [0:63];
    integer rr, cc, kk;
    integer acc;
    integer a_val;
    integer b_val;
    integer c_val;
    begin
        for (rr = 0; rr < 8; rr = rr + 1) begin
            for (cc = 0; cc < 8; cc = cc + 1) begin
                c_val = conv_c_lane_from_bus(C_dtype, C_bus, rr*8+cc);
                acc = c_val;
                for (kk = 0; kk < 8; kk = kk + 1) begin
                    a_val = conv_ab_lane_from_bus(A_dtype, A_bus, rr*8+kk);
                    b_val = conv_ab_lane_from_bus(B_dtype, B_bus, kk*8+cc);
                    acc = acc + a_val * b_val;
                end
                REF[rr*8+cc] = acc[7:0];
            end
        end
    end
endtask

task calc_sparse_ref_fp8;
    input [MATRIX_BUS-1:0] A_comp_bus;
    input [META_WIDTH-1:0] META_bus;
    input [2:0] mode;
    input [MATRIX_BUS-1:0] B_bus;
    input [1:0] B_dtype;
    input [MATRIX_BUS-1:0] C_bus;
    input [1:0] C_dtype;
    output reg signed [7:0] REF [0:63];
    integer rr, cc, kk;
    integer acc;
    integer a_val;
    integer b_val;
    integer c_val;
    begin
        for (rr = 0; rr < 8; rr = rr + 1) begin
            for (cc = 0; cc < 8; cc = cc + 1) begin
                c_val = conv_c_lane_from_bus(C_dtype, C_bus, rr*8+cc);
                acc = c_val;
                for (kk = 0; kk < 8; kk = kk + 1) begin
                    a_val = sparse_a_lane_from_bus_fp8(A_comp_bus, META_bus, mode, rr*8+kk);
                    b_val = conv_ab_lane_from_bus(B_dtype, B_bus, kk*8+cc);
                    acc = acc + a_val * b_val;
                end
                REF[rr*8+cc] = acc[7:0];
            end
        end
    end
endtask

// =====================================================================
// Drive / Check
// =====================================================================
task drive_once;
    input [MATRIX_BUS-1:0] A_bus;
    input [MATRIX_BUS-1:0] B_bus;
    input [MATRIX_BUS-1:0] C_bus;
    input [META_WIDTH-1:0] META_bus;
    input [2:0] mode;
    input compressed;
    input mvalid;
    begin
        @(negedge clk);
        s_axis_tdata_a <= A_bus;
        s_axis_tdata_b <= B_bus;
        s_axis_tdata_c <= C_bus;
        s_axis_tmeta_a <= META_bus;
        sparse_mode_sel_i <= mode;
        a_is_compressed_i <= compressed;
        meta_valid_i      <= mvalid;
        s_axis_tvalid_a   <= 1'b1;
        s_axis_tvalid_b   <= 1'b1;
        s_axis_tvalid_c   <= 1'b1;
        mma_start         <= 1'b1;
        @(negedge clk);
        s_axis_tvalid_a   <= 1'b0;
        s_axis_tvalid_b   <= 1'b0;
        s_axis_tvalid_c   <= 1'b0;
        mma_start         <= 1'b0;
        sparse_mode_sel_i <= `SPARSE_MODE_DENSE;
        a_is_compressed_i <= 1'b0;
        meta_valid_i      <= 1'b0;
        s_axis_tmeta_a    <= {META_WIDTH{1'b0}};
    end
endtask

task check_output;
    input reg signed [7:0] REF [0:63];
    input [255:0] tag;
    integer idx;
    integer errors;
    reg signed [7:0] got;
    begin
        errors = 0;
        timeout_counter = 0;
        while ((m_axis_tvalid_d !== 1'b1) && (timeout_counter < 300)) begin
            @(posedge clk);
            timeout_counter = timeout_counter + 1;
        end
        if (m_axis_tvalid_d !== 1'b1) begin
            $display("[%0s] TIMEOUT waiting for output", tag);
            total_fail = total_fail + 1;
            total_tests = total_tests + 1;
        end else begin
            #1;
            for (idx = 0; idx < 64; idx = idx + 1) begin
                got = wb_wvd_rd_o[idx*8 +: 8];
                if (got !== REF[idx]) begin
                    errors = errors + 1;
                    if (errors <= 16)
                        $display("[%0s] mismatch idx=%0d got=%0d exp=%0d", tag, idx, got, REF[idx]);
                end
            end
            total_tests = total_tests + 1;
            if (errors == 0) begin
                total_pass = total_pass + 1;
                $display("[%0s] PASS", tag);
            end else begin
                total_fail = total_fail + 1;
                $display("[%0s] FAIL errors=%0d", tag, errors);
            end
            @(negedge clk);
        end
    end
endtask

// =====================================================================
// Dense scalar case suite (keeps old tb's spirit: 3 numeric cases × 12 type combos)
// =====================================================================
task run_dense_scalar_case;
    input [1:0] a_dtype_i;
    input [1:0] b_dtype_i;
    input [1:0] c_dtype_i;
    input [15:0] a_raw;
    input [15:0] b_raw;
    input [31:0] c_raw;
    input [255:0] tag;
    reg [MATRIX_BUS-1:0] A_bus;
    reg [MATRIX_BUS-1:0] B_bus;
    reg [MATRIX_BUS-1:0] C_bus;
    reg signed [7:0] REF [0:63];
    begin
        a_type = a_dtype_i;
        b_type = b_dtype_i;
        c_type = c_dtype_i;
        d_type = d_fp8;
        build_uniform_ab_bus(a_dtype_i, a_raw, A_bus);
        build_uniform_ab_bus(b_dtype_i, b_raw, B_bus);
        build_uniform_c_bus(c_dtype_i, c_raw, C_bus);
        calc_dense_ref_from_buses(A_bus, a_dtype_i, B_bus, b_dtype_i, C_bus, c_dtype_i, REF);
        drive_once(A_bus, B_bus, C_bus, {META_WIDTH{1'b0}}, `SPARSE_MODE_DENSE, 1'b0, 1'b0);
        check_output(REF, tag);
        repeat (4) @(posedge clk);
    end
endtask


task run_dense_asym_case;
    input [1:0] a_dtype_i;
    input [1:0] b_dtype_i;
    input [1:0] c_dtype_i;
    input integer a_pat;
    input integer b_pat;
    input integer c_pat;
    input [255:0] tag;
    reg [MATRIX_BUS-1:0] A_bus;
    reg [MATRIX_BUS-1:0] B_bus;
    reg [MATRIX_BUS-1:0] C_bus;
    reg signed [7:0] REF [0:63];
    begin
        a_type = a_dtype_i;
        b_type = b_dtype_i;
        c_type = c_dtype_i;
        d_type = d_fp8;
        build_asym_ab_bus(a_dtype_i, a_pat, A_bus);
        build_asym_ab_bus(b_dtype_i, b_pat, B_bus);
        build_asym_c_bus(c_dtype_i, c_pat, C_bus);
        calc_dense_ref_from_buses(A_bus, a_dtype_i, B_bus, b_dtype_i, C_bus, c_dtype_i, REF);
        drive_once(A_bus, B_bus, C_bus, {META_WIDTH{1'b0}}, `SPARSE_MODE_DENSE, 1'b0, 1'b0);
        check_output(REF, tag);
        repeat (4) @(posedge clk);
    end
endtask

// =====================================================================
// Sparse deterministic matrix suite (FP8 A path)
// =====================================================================
reg signed [7:0] A_sparse_2to4 [0:63];
reg signed [7:0] A_sparse_1to4 [0:63];
reg signed [7:0] A_comp_2to4 [0:63];
reg signed [7:0] A_comp_1to4 [0:63];
reg signed [7:0] B_dense_mat [0:63];
reg signed [7:0] C_zero_mat [0:63];
reg [7:0] META_2to4 [0:15];
reg [7:0] META_1to4 [0:15];

task init_sparse_mats;
    integer r, c;
    begin
        for (r = 0; r < 64; r = r + 1) begin
            A_sparse_2to4[r] = 0;
            A_sparse_1to4[r] = 0;
            A_comp_2to4[r]   = 0;
            A_comp_1to4[r]   = 0;
            B_dense_mat[r]   = 0;
            C_zero_mat[r]    = 0;
        end
        for (r = 0; r < 16; r = r + 1) begin
            META_2to4[r] = 8'h00;
            META_1to4[r] = 8'h00;
        end

        // 2:4 sparse A: each row keeps positions 0,2 and 4,6.
        for (r = 0; r < 8; r = r + 1) begin
            A_sparse_2to4[r*8+0] = r + 1;
            A_sparse_2to4[r*8+2] = r + 2;
            A_sparse_2to4[r*8+4] = r + 3;
            A_sparse_2to4[r*8+6] = r + 4;
            A_comp_2to4[r*8+0]   = A_sparse_2to4[r*8+0];
            A_comp_2to4[r*8+1]   = A_sparse_2to4[r*8+2];
            A_comp_2to4[r*8+2]   = A_sparse_2to4[r*8+4];
            A_comp_2to4[r*8+3]   = A_sparse_2to4[r*8+6];
            META_2to4[r*2+0] = 8'b00001000; // idx0=0 idx1=2
            META_2to4[r*2+1] = 8'b00001000; // idx0=0 idx1=2
        end

        // 1:4 sparse A: each group keeps position 1.
        for (r = 0; r < 8; r = r + 1) begin
            A_sparse_1to4[r*8+1] = r + 5;
            A_sparse_1to4[r*8+5] = r + 6;
            A_comp_1to4[r*8+0]   = A_sparse_1to4[r*8+1];
            A_comp_1to4[r*8+1]   = A_sparse_1to4[r*8+5];
            META_1to4[r*2+0] = 8'b00000001; // idx0=1
            META_1to4[r*2+1] = 8'b00000001; // idx0=1
        end

        for (r = 0; r < 8; r = r + 1)
            for (c = 0; c < 8; c = c + 1) begin
                B_dense_mat[r*8+c] = (r*3 - c*2 + 9);
                C_zero_mat[r*8+c]  = ((r + c) & 1) ? -(r + 1) : (c + 1);
            end
    end
endtask

task run_sparse_case;
    input [2:0] mode;
    input [1:0] b_dtype_i;
    input [1:0] c_dtype_i;
    input [255:0] tag;
    reg [MATRIX_BUS-1:0] A_bus;
    reg [MATRIX_BUS-1:0] B_bus;
    reg [MATRIX_BUS-1:0] C_bus;
    reg [META_WIDTH-1:0] META_bus;
    reg signed [7:0] REF [0:63];
    begin
        a_type = ab_FP8E4M3;
        b_type = b_dtype_i;
        c_type = c_dtype_i;
        d_type = d_fp8;
        if (mode == `SPARSE_MODE_2TO4) begin
            pack_matrix8(A_comp_2to4, A_bus);
            pack_meta(META_2to4, META_bus);
        end else begin
            pack_matrix8(A_comp_1to4, A_bus);
            pack_meta(META_1to4, META_bus);
        end
        // For B/C, use deterministic integer matrices. B is packed as FP8 bytes;
        // c_type variation still exercises the converter path because C_bus is zero.
        pack_matrix8(B_dense_mat, B_bus);
        C_bus = {MATRIX_BUS{1'b0}};
        calc_sparse_ref_fp8(A_bus, META_bus, mode, B_bus, b_dtype_i, C_bus, c_dtype_i, REF);
        drive_once(A_bus, B_bus, C_bus, META_bus, mode, 1'b1, 1'b1);
        check_output(REF, tag);
        repeat (4) @(posedge clk);
    end
endtask


task run_sparse_case_asym;
    input [2:0] mode;
    input [1:0] b_dtype_i;
    input [1:0] c_dtype_i;
    input integer b_pat;
    input integer c_pat;
    input [255:0] tag;
    reg [MATRIX_BUS-1:0] A_bus;
    reg [MATRIX_BUS-1:0] B_bus;
    reg [MATRIX_BUS-1:0] C_bus;
    reg [META_WIDTH-1:0] META_bus;
    reg signed [7:0] REF [0:63];
    begin
        a_type = ab_FP8E4M3;
        b_type = b_dtype_i;
        c_type = c_dtype_i;
        d_type = d_fp8;
        if (mode == `SPARSE_MODE_2TO4) begin
            pack_matrix8(A_comp_2to4, A_bus);
            pack_meta(META_2to4, META_bus);
        end else begin
            pack_matrix8(A_comp_1to4, A_bus);
            pack_meta(META_1to4, META_bus);
        end
        build_asym_ab_bus(b_dtype_i, b_pat, B_bus);
        build_asym_c_bus(c_dtype_i, c_pat, C_bus);
        calc_sparse_ref_fp8(A_bus, META_bus, mode, B_bus, b_dtype_i, C_bus, c_dtype_i, REF);
        drive_once(A_bus, B_bus, C_bus, META_bus, mode, 1'b1, 1'b1);
        check_output(REF, tag);
        repeat (4) @(posedge clk);
    end
endtask

// =====================================================================
// Main Test Process
// =====================================================================
initial begin
    reg [1:0] ab_types [0:2];
    reg [1:0] c_types [0:3];
    reg [15:0] a_case [0:2][0:2];
    reg [15:0] b_case [0:2][0:2];
    reg [31:0] c_case [0:3][0:2];
    integer ai, ci, cs;

    sys_init();
    init_sparse_mats();

    // type lists
    ab_types[0] = ab_FP8E4M3;
    ab_types[1] = ab_FP8E5M2;
    ab_types[2] = ab_FP16;
    c_types[0] = c_FP8E4M3;
    c_types[1] = c_FP8E5M2;
    c_types[2] = c_FP16;
    c_types[3] = c_FP32;

    // case 0/1/2 for each a/b type, matching the old tb's raw encodings.
    // ab_FP8E4M3
    a_case[0][0] = 16'h0038; b_case[0][0] = 16'h0038;
    a_case[0][1] = 16'h003C; b_case[0][1] = 16'h0040;
    a_case[0][2] = 16'h00C0; b_case[0][2] = 16'h003C;
    // ab_FP8E5M2
    a_case[1][0] = 16'h003C; b_case[1][0] = 16'h003C;
    a_case[1][1] = 16'h003E; b_case[1][1] = 16'h0040;
    a_case[1][2] = 16'h00C0; b_case[1][2] = 16'h003E;
    // ab_FP16
    a_case[2][0] = 16'h3C00; b_case[2][0] = 16'h3C00;
    a_case[2][1] = 16'h3E00; b_case[2][1] = 16'h4000;
    a_case[2][2] = 16'hC000; b_case[2][2] = 16'h3E00;

    // c raw values per c_type and case, matching old tb's chosen patterns.
    c_case[0][0] = 32'h00000040; c_case[0][1] = 32'h00000042; c_case[0][2] = 32'h0000004C;
    c_case[1][0] = 32'h00000040; c_case[1][1] = 32'h00000041; c_case[1][2] = 32'h00000046;
    c_case[2][0] = 32'h00004000; c_case[2][1] = 32'h00004100; c_case[2][2] = 32'h00004600;
    c_case[3][0] = 32'h40000000; c_case[3][1] = 32'h40200000; c_case[3][2] = 32'h40C00000;

    @(posedge rst_n);
    repeat(5) @(posedge clk);

    $display("\n================ Dense overall smoke suite ================");
    for (ai = 0; ai < 3; ai = ai + 1) begin
        for (ci = 0; ci < 4; ci = ci + 1) begin
            for (cs = 0; cs < 3; cs = cs + 1) begin
                $display("[DENSE] ab_type=%0d c_type=%0d case=%0d", ab_types[ai], c_types[ci], cs);
                run_dense_scalar_case(ab_types[ai], ab_types[ai], c_types[ci],
                                      a_case[ai][cs], b_case[ai][cs], c_case[ci][cs],
                                      "dense_case");
            end
        end
    end

    $display("\n================ Dense mixed-precision A/B suite ================");
    // Mixed-precision means A and B use different input formats.
    // These cases specifically verify A!=B datatype handling.
    run_dense_scalar_case(ab_FP8E4M3, ab_FP16,    c_FP16,    16'h003C, 16'h3C00, 32'h00004000, "dense_mixed_Afp8e4m3_Bfp16_Cfp16_case0");
    run_dense_scalar_case(ab_FP8E4M3, ab_FP16,    c_FP32,    16'h00C0, 16'h4000, 32'h40000000, "dense_mixed_Afp8e4m3_Bfp16_Cfp32_case1");
    run_dense_scalar_case(ab_FP8E5M2, ab_FP16,    c_FP8E5M2, 16'h003E, 16'h3E00, 32'h00000041, "dense_mixed_Afp8e5m2_Bfp16_Cfp8e5m2_case2");
    run_dense_scalar_case(ab_FP16,    ab_FP8E4M3, c_FP16,    16'h3C00, 16'h0038, 32'h00004100, "dense_mixed_Afp16_Bfp8e4m3_Cfp16_case3");
    run_dense_scalar_case(ab_FP16,    ab_FP8E5M2, c_FP32,    16'hC000, 16'h003E, 32'h40200000, "dense_mixed_Afp16_Bfp8e5m2_Cfp32_case4");
    run_dense_scalar_case(ab_FP8E5M2, ab_FP8E4M3, c_FP8E4M3, 16'h003C, 16'h0038, 32'h00000042, "dense_mixed_Afp8e5m2_Bfp8e4m3_Cfp8e4m3_case5");
    run_dense_scalar_case(ab_FP8E4M3, ab_FP8E5M2, c_FP8E4M3, 16'h0038, 16'h003C, 32'h0000004C, "dense_mixed_Afp8e4m3_Bfp8e5m2_Cfp8e4m3_case6");

    $display("\n================ Dense asymmetric mixed-precision suite ================");
    run_dense_asym_case(ab_FP8E4M3, ab_FP16,    c_FP32,    0, 1, 1, "dense_asym_mixed_Afp8e4m3_Bfp16_Cfp32");
    run_dense_asym_case(ab_FP16,    ab_FP8E5M2, c_FP16,    1, 0, 0, "dense_asym_mixed_Afp16_Bfp8e5m2_Cfp16");
    run_dense_asym_case(ab_FP8E5M2, ab_FP8E4M3, c_FP8E5M2, 0, 1, 0, "dense_asym_mixed_Afp8e5m2_Bfp8e4m3_Cfp8e5m2");
    run_dense_asym_case(ab_FP8E4M3, ab_FP8E5M2, c_FP16,    1, 0, 1, "dense_asym_mixed_Afp8e4m3_Bfp8e5m2_Cfp16");

    $display("\n================ Sparse overall suite ================");
    run_sparse_case(`SPARSE_MODE_2TO4, ab_FP8E4M3, c_FP8E5M2, "sparse_2to4_fp8e4m3");
    run_sparse_case(`SPARSE_MODE_1TO4, ab_FP8E4M3, c_FP8E5M2, "sparse_1to4_fp8e4m3");
    run_sparse_case(`SPARSE_MODE_2TO4, ab_FP8E5M2, c_FP8E4M3, "sparse_2to4_fp8e5m2");
    run_sparse_case(`SPARSE_MODE_1TO4, ab_FP8E5M2, c_FP8E4M3, "sparse_1to4_fp8e5m2");
    run_sparse_case(`SPARSE_MODE_2TO4, ab_FP8E4M3, c_FP16,    "sparse_2to4_c_fp16");
    run_sparse_case(`SPARSE_MODE_1TO4, ab_FP8E4M3, c_FP32,    "sparse_1to4_c_fp32");

    $display("\n================ Sparse asymmetric suite ================");
    run_sparse_case_asym(`SPARSE_MODE_2TO4, ab_FP8E4M3, c_FP8E4M3, 0, 0, "sparse_asym_2to4_fp8");
    run_sparse_case_asym(`SPARSE_MODE_1TO4, ab_FP8E5M2, c_FP16,    1, 0, "sparse_asym_1to4_cfp16");
    run_sparse_case_asym(`SPARSE_MODE_2TO4, ab_FP8E5M2, c_FP32,    0, 1, "sparse_asym_2to4_cfp32");
    run_sparse_case_asym(`SPARSE_MODE_1TO4, ab_FP8E4M3, c_FP8E5M2, 1, 1, "sparse_asym_1to4_fp8mix");

    $display("\n=======================================================");
    $display("TB SUMMARY: total=%0d pass=%0d fail=%0d", total_tests, total_pass, total_fail);
    if (total_fail == 0)
        $display("ALL TESTS PASSED.");
    else
        $display("SOME TESTS FAILED.");
    $finish;
end

// =====================================================================
// Waveform Dump
// =====================================================================
initial begin
    `ifdef DUMP_FSDB
        $fsdbDumpfile("tb_tensor_core_sparse.fsdb");
        $fsdbDumpvars(0, tb_tensor_core);
        $fsdbDumpMDA();
    `else
        $dumpfile("tb_tensor_core_sparse.vcd");
        $dumpvars(0, tb_tensor_core);
    `endif
end

endmodule

