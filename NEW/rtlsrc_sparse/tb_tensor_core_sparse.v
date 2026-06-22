`timescale 1ns/1ns
`include "define.v"

module tb_tensor_core_sparse;
    localparam SHAPE_M = 8;
    localparam SHAPE_K = 8;
    localparam SHAPE_N = 8;
    localparam META_WIDTH = SHAPE_M*(SHAPE_K/4)*`SPARSE_META_BITS_PER_GROUP;

    reg clk;
    reg rst_n;
    reg [`MATRIX_BUS_WIDTH-1:0] s_axis_tdata_a;
    reg [`MATRIX_BUS_WIDTH-1:0] s_axis_tdata_b;
    reg [`MATRIX_BUS_WIDTH-1:0] s_axis_tdata_c;
    reg [2:0] sparse_mode_sel_i;
    reg a_is_compressed_i;
    reg meta_valid_i;
    reg [META_WIDTH-1:0] s_axis_tmeta_a;
    reg [`REGIDX_WIDTH+`REGEXT_WIDTH-1:0] ctrl_reg_idxw_i;
    reg [`DEPTH_WARP-1:0] ctrl_wid_i;
    reg [2:0] rm_i;
    reg mma_start_i;
    wire mma_ready;
    reg [1:0] a_type_i;
    reg [1:0] b_type_i;
    reg [1:0] c_type_i;
    reg d_type_i;
    reg s_axis_tvalid_a;
    wire s_axis_tready_a;
    reg s_axis_tvalid_b;
    wire s_axis_tready_b;
    reg s_axis_tvalid_c;
    wire s_axis_tready_c;
    reg m_axis_tready_d;
    wire m_axis_tvalid_d;
    wire m_axis_tlast_d;
    wire [SHAPE_M*SHAPE_N*`XLEN_FP8-1:0] wb_wvd_rd_o;

    reg signed [7:0] A_dense [0:63];
    reg signed [7:0] A_sparse_2to4 [0:63];
    reg signed [7:0] A_sparse_1to4 [0:63];
    reg signed [7:0] B_dense [0:63];
    reg signed [7:0] C_dense [0:63];
    reg signed [7:0] A_comp_2to4 [0:63];
    reg signed [7:0] A_comp_1to4 [0:63];
    reg [7:0] META_2to4 [0:15];
    reg [7:0] META_1to4 [0:15];
    reg signed [15:0] REF_dense [0:63];
    reg signed [15:0] REF_sparse_2to4 [0:63];
    reg signed [15:0] REF_sparse_1to4 [0:63];
    integer i, j, k;
    integer errors;

    tensor_core dut (
        .clk(clk),
        .rst_n(rst_n),
        .s_axis_tdata_a(s_axis_tdata_a),
        .s_axis_tdata_b(s_axis_tdata_b),
        .s_axis_tdata_c(s_axis_tdata_c),
        .sparse_mode_sel_i(sparse_mode_sel_i),
        .a_is_compressed_i(a_is_compressed_i),
        .meta_valid_i(meta_valid_i),
        .s_axis_tmeta_a(s_axis_tmeta_a),
        .ctrl_reg_idxw_i(ctrl_reg_idxw_i),
        .ctrl_wid_i(ctrl_wid_i),
        .rm_i(rm_i),
        .mma_start_i(mma_start_i),
        .mma_ready(mma_ready),
        .a_type_i(a_type_i),
        .b_type_i(b_type_i),
        .c_type_i(c_type_i),
        .d_type_i(d_type_i),
        .s_axis_tvalid_a(s_axis_tvalid_a),
        .s_axis_tready_a(s_axis_tready_a),
        .s_axis_tvalid_b(s_axis_tvalid_b),
        .s_axis_tready_b(s_axis_tready_b),
        .s_axis_tvalid_c(s_axis_tvalid_c),
        .s_axis_tready_c(s_axis_tready_c),
        .m_axis_tready_d(m_axis_tready_d),
        .m_axis_tvalid_d(m_axis_tvalid_d),
        .m_axis_tlast_d(m_axis_tlast_d),
        .wb_wvd_rd_o(wb_wvd_rd_o)
    );

    always #5 clk = ~clk;

    task pack_matrix8;
        input integer dummy;
        input reg signed [7:0] arr [0:63];
        output reg [`MATRIX_BUS_WIDTH-1:0] packed_bus;
        integer idx;
        begin
            packed_bus = {`MATRIX_BUS_WIDTH{1'b0}};
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

    task calc_ref;
        input reg signed [7:0] A [0:63];
        output reg signed [15:0] R [0:63];
        integer rr, cc, kk;
        integer sum;
        begin
            for (rr = 0; rr < 8; rr = rr + 1) begin
                for (cc = 0; cc < 8; cc = cc + 1) begin
                    sum = C_dense[rr*8+cc];
                    for (kk = 0; kk < 8; kk = kk + 1)
                        sum = sum + A[rr*8+kk] * B_dense[kk*8+cc];
                    R[rr*8+cc] = sum[15:0];
                end
            end
        end
    endtask

    task drive_once;
        input [`MATRIX_BUS_WIDTH-1:0] A_pack;
        input [`MATRIX_BUS_WIDTH-1:0] B_pack;
        input [`MATRIX_BUS_WIDTH-1:0] C_pack;
        input [META_WIDTH-1:0] META_pack;
        input [2:0] mode;
        input compressed;
        input mvalid;
        begin
            @(negedge clk);
            s_axis_tdata_a  <= A_pack;
            s_axis_tdata_b  <= B_pack;
            s_axis_tdata_c  <= C_pack;
            s_axis_tmeta_a  <= META_pack;
            sparse_mode_sel_i <= mode;
            a_is_compressed_i <= compressed;
            meta_valid_i <= mvalid;
            s_axis_tvalid_a <= 1'b1;
            s_axis_tvalid_b <= 1'b1;
            s_axis_tvalid_c <= 1'b1;
            mma_start_i     <= 1'b1;
            @(negedge clk);
            s_axis_tvalid_a <= 1'b0;
            s_axis_tvalid_b <= 1'b0;
            s_axis_tvalid_c <= 1'b0;
            mma_start_i     <= 1'b0;
            sparse_mode_sel_i <= `SPARSE_MODE_DENSE;
            a_is_compressed_i <= 1'b0;
            meta_valid_i <= 1'b0;
            s_axis_tmeta_a <= {META_WIDTH{1'b0}};
        end
    endtask

    task check_output;
        input reg signed [15:0] REF [0:63];
        input [255:0] tag;
        integer idx;
        reg signed [7:0] got;
        begin
            errors = 0;
            wait (m_axis_tvalid_d === 1'b1);
            #1;
            for (idx = 0; idx < 64; idx = idx + 1) begin
                got = wb_wvd_rd_o[idx*8 +: 8];
                if (got !== REF[idx][7:0]) begin
                    errors = errors + 1;
                    $display("[%0s] mismatch idx=%0d got=%0d exp=%0d", tag, idx, got, REF[idx][7:0]);
                end
            end
            if (errors == 0)
                $display("[%0s] PASS", tag);
            else begin
                $display("[%0s] FAIL errors=%0d", tag, errors);
                $fatal(1);
            end
            @(negedge clk);
        end
    endtask

    initial begin
        #2000;
        $display("TIMEOUT: m_axis_tvalid_d never asserted, simulation is stuck waiting for output.");
        $fatal(1);
    end

    initial begin
        clk = 0;
        rst_n = 0;
        s_axis_tdata_a = 0;
        s_axis_tdata_b = 0;
        s_axis_tdata_c = 0;
        sparse_mode_sel_i = `SPARSE_MODE_DENSE;
        a_is_compressed_i = 0;
        meta_valid_i = 0;
        s_axis_tmeta_a = 0;
        ctrl_reg_idxw_i = 0;
        ctrl_wid_i = 0;
        rm_i = 3'b000;
        mma_start_i = 0;
        a_type_i = 2'b00;
        b_type_i = 2'b00;
        c_type_i = 2'b00;
        d_type_i = 1'b0;
        s_axis_tvalid_a = 0;
        s_axis_tvalid_b = 0;
        s_axis_tvalid_c = 0;
        m_axis_tready_d = 1'b1;

        for (i = 0; i < 64; i = i + 1) begin
            A_dense[i] = 0;
            A_sparse_2to4[i] = 0;
            A_sparse_1to4[i] = 0;
            B_dense[i] = 0;
            C_dense[i] = 0;
            A_comp_2to4[i] = 0;
            A_comp_1to4[i] = 0;
        end
        for (i = 0; i < 16; i = i + 1) begin
            META_2to4[i] = 8'h00;
            META_1to4[i] = 8'h00;
        end

        // Build a deterministic 2:4 sparse A. Each 4-lane group keeps positions 0 and 2.
        for (i = 0; i < 8; i = i + 1) begin
            A_sparse_2to4[i*8+0] = i + 1;
            A_sparse_2to4[i*8+2] = i + 2;
            A_sparse_2to4[i*8+4] = i + 3;
            A_sparse_2to4[i*8+6] = i + 4;
            A_comp_2to4[i*8+0]   = A_sparse_2to4[i*8+0];
            A_comp_2to4[i*8+1]   = A_sparse_2to4[i*8+2];
            A_comp_2to4[i*8+2]   = A_sparse_2to4[i*8+4];
            A_comp_2to4[i*8+3]   = A_sparse_2to4[i*8+6];
            META_2to4[i*2+0] = 8'b00001000; // idx0=0 idx1=2
            META_2to4[i*2+1] = 8'b00001000; // idx0=0 idx1=2 for second group
        end

        // Build a deterministic 1:4 sparse A. Each group keeps only position 1.
        for (i = 0; i < 8; i = i + 1) begin
            A_sparse_1to4[i*8+1] = i + 5;
            A_sparse_1to4[i*8+5] = i + 6;
            A_comp_1to4[i*8+0]   = A_sparse_1to4[i*8+1];
            A_comp_1to4[i*8+1]   = A_sparse_1to4[i*8+5];
            META_1to4[i*2+0] = 8'b00000001; // idx0=1
            META_1to4[i*2+1] = 8'b00000001; // idx0=1
        end

        // Dense B and C.
        for (i = 0; i < 8; i = i + 1)
            for (j = 0; j < 8; j = j + 1) begin
                B_dense[i*8+j] = (i == j) ? 1 : (j + 1);
                C_dense[i*8+j] = j;
            end

        calc_ref(A_sparse_2to4, REF_dense);
        calc_ref(A_sparse_2to4, REF_sparse_2to4);
        calc_ref(A_sparse_1to4, REF_sparse_1to4);

        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        // Dense reference run using already sparse-shaped dense A.
        pack_matrix8(0, A_sparse_2to4, s_axis_tdata_a);
        pack_matrix8(0, B_dense, s_axis_tdata_b);
        pack_matrix8(0, C_dense, s_axis_tdata_c);
        drive_once(s_axis_tdata_a, s_axis_tdata_b, s_axis_tdata_c, {META_WIDTH{1'b0}}, `SPARSE_MODE_DENSE, 1'b0, 1'b0);
        check_output(REF_dense, "dense_baseline");

        // 2:4 sparse run.
        pack_matrix8(0, A_comp_2to4, s_axis_tdata_a);
        pack_matrix8(0, B_dense, s_axis_tdata_b);
        pack_matrix8(0, C_dense, s_axis_tdata_c);
        pack_meta(META_2to4, s_axis_tmeta_a);
        drive_once(s_axis_tdata_a, s_axis_tdata_b, s_axis_tdata_c, s_axis_tmeta_a, `SPARSE_MODE_2TO4, 1'b1, 1'b1);
        check_output(REF_sparse_2to4, "sparse_2to4");

        // 1:4 sparse run.
        pack_matrix8(0, A_comp_1to4, s_axis_tdata_a);
        pack_matrix8(0, B_dense, s_axis_tdata_b);
        pack_matrix8(0, C_dense, s_axis_tdata_c);
        pack_meta(META_1to4, s_axis_tmeta_a);
        drive_once(s_axis_tdata_a, s_axis_tdata_b, s_axis_tdata_c, s_axis_tmeta_a, `SPARSE_MODE_1TO4, 1'b1, 1'b1);
        check_output(REF_sparse_1to4, "sparse_1to4");

        $display("All tests passed.");
        $finish;
    end
endmodule

