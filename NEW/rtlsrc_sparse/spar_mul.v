`timescale 1ns/1ns
`include "define.v"

module spar_mul #(
    parameter SHAPE_M = 8,
    parameter SHAPE_K = 8,
    parameter ELEM_WIDTH = `XLEN_FP9E5M3,
    parameter META_BITS_PER_GROUP = `SPARSE_META_BITS_PER_GROUP
)(
    input  [2:0] sparse_mode_sel_i,
    input  [SHAPE_M*SHAPE_K*ELEM_WIDTH-1:0] a_comp_i,
    input  [SHAPE_M*(SHAPE_K/4)*META_BITS_PER_GROUP-1:0] meta_i,
    output reg [SHAPE_M*SHAPE_K*ELEM_WIDTH-1:0] a_align_o,
    output reg [SHAPE_M*SHAPE_K-1:0] lane_valid_o
);

integer r, g;
integer row_base;
integer payload_base;
integer meta_base;
reg [1:0] idx0;
reg [1:0] idx1;
localparam GROUPS_PER_ROW = SHAPE_K/4;

always @(*) begin
    a_align_o = {(SHAPE_M*SHAPE_K*ELEM_WIDTH){1'b0}};
    lane_valid_o = {(SHAPE_M*SHAPE_K){1'b0}};

    if (sparse_mode_sel_i == `SPARSE_MODE_DENSE) begin
        a_align_o = a_comp_i;
        lane_valid_o = {(SHAPE_M*SHAPE_K){1'b1}};
    end else begin
        for (r = 0; r < SHAPE_M; r = r + 1) begin
            row_base = r * SHAPE_K;
            if (sparse_mode_sel_i == `SPARSE_MODE_2TO4) begin
                for (g = 0; g < GROUPS_PER_ROW; g = g + 1) begin
                    payload_base = row_base + (g * 2);
                    meta_base = (r * GROUPS_PER_ROW + g) * META_BITS_PER_GROUP;
                    idx0 = meta_i[meta_base +: 2];
                    idx1 = meta_i[meta_base+2 +: 2];
                    a_align_o[((row_base + g*4 + idx0) * ELEM_WIDTH) +: ELEM_WIDTH] = a_comp_i[(payload_base * ELEM_WIDTH) +: ELEM_WIDTH];
                    a_align_o[((row_base + g*4 + idx1) * ELEM_WIDTH) +: ELEM_WIDTH] = a_comp_i[((payload_base + 1) * ELEM_WIDTH) +: ELEM_WIDTH];
                    lane_valid_o[row_base + g*4 + idx0] = 1'b1;
                    lane_valid_o[row_base + g*4 + idx1] = 1'b1;
                end
            end else if (sparse_mode_sel_i == `SPARSE_MODE_1TO4) begin
                for (g = 0; g < GROUPS_PER_ROW; g = g + 1) begin
                    payload_base = row_base + g;
                    meta_base = (r * GROUPS_PER_ROW + g) * META_BITS_PER_GROUP;
                    idx0 = meta_i[meta_base +: 2];
                    a_align_o[((row_base + g*4 + idx0) * ELEM_WIDTH) +: ELEM_WIDTH] = a_comp_i[(payload_base * ELEM_WIDTH) +: ELEM_WIDTH];
                    lane_valid_o[row_base + g*4 + idx0] = 1'b1;
                end
            end
        end
    end
end

endmodule

