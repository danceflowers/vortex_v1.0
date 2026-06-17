`timescale 1ns / 1ps

module spar_mul #(
    parameter integer ELEM_WIDTH = 9,
    parameter integer BLOCKS = 4,
    parameter integer LANES_PER_BLOCK = 2,
    parameter integer META_WIDTH = 64
) (
    input  wire [2:0] sparse_mode_sel_i,
    input  wire [BLOCKS*LANES_PER_BLOCK*ELEM_WIDTH-1:0] a_comp_i,
    input  wire [BLOCKS*4*ELEM_WIDTH-1:0]               b_dense_i,
    input  wire [META_WIDTH-1:0]                        meta_i,

    output reg  [BLOCKS*LANES_PER_BLOCK-1:0]            lane_valid_o,
    output reg  [BLOCKS*LANES_PER_BLOCK*ELEM_WIDTH-1:0] a_sel_o,
    output reg  [BLOCKS*LANES_PER_BLOCK*ELEM_WIDTH-1:0] b_sel_o,
    output reg  [BLOCKS*LANES_PER_BLOCK*ELEM_WIDTH*ELEM_WIDTH-1:0] pp_o
);

    localparam integer LANES = BLOCKS * LANES_PER_BLOCK;

    integer blk;
    integer lane;
    integer bit_a;
    integer bit_b;
    reg [1:0] idx0;
    reg [1:0] idx1;
    reg [ELEM_WIDTH-1:0] a_lane;
    reg [ELEM_WIDTH-1:0] b_lane;

    function [ELEM_WIDTH-1:0] pick_b_elem;
        input integer block_id;
        input [1:0] idx;
        begin
            case (idx)
                2'd0: pick_b_elem = b_dense_i[(block_id*4+0)*ELEM_WIDTH +: ELEM_WIDTH];
                2'd1: pick_b_elem = b_dense_i[(block_id*4+1)*ELEM_WIDTH +: ELEM_WIDTH];
                2'd2: pick_b_elem = b_dense_i[(block_id*4+2)*ELEM_WIDTH +: ELEM_WIDTH];
                default: pick_b_elem = b_dense_i[(block_id*4+3)*ELEM_WIDTH +: ELEM_WIDTH];
            endcase
        end
    endfunction

    always @(*) begin
        lane_valid_o = {LANES{1'b0}};
        a_sel_o      = {(LANES*ELEM_WIDTH){1'b0}};
        b_sel_o      = {(LANES*ELEM_WIDTH){1'b0}};
        pp_o         = {(LANES*ELEM_WIDTH*ELEM_WIDTH){1'b0}};

        if (sparse_mode_sel_i == 3'b001) begin
            for (blk = 0; blk < BLOCKS; blk = blk + 1) begin
                idx0 = meta_i[blk*4 +: 2];
                idx1 = meta_i[blk*4 + 2 +: 2];

                for (lane = 0; lane < LANES_PER_BLOCK; lane = lane + 1) begin
                    lane_valid_o[blk*LANES_PER_BLOCK + lane] = 1'b1;
                    if (lane == 0) begin
                        a_lane = a_comp_i[(blk*LANES_PER_BLOCK + 0)*ELEM_WIDTH +: ELEM_WIDTH];
                        b_lane = pick_b_elem(blk, idx0);
                    end else begin
                        a_lane = a_comp_i[(blk*LANES_PER_BLOCK + 1)*ELEM_WIDTH +: ELEM_WIDTH];
                        b_lane = pick_b_elem(blk, idx1);
                    end

                    a_sel_o[(blk*LANES_PER_BLOCK + lane)*ELEM_WIDTH +: ELEM_WIDTH] = a_lane;
                    b_sel_o[(blk*LANES_PER_BLOCK + lane)*ELEM_WIDTH +: ELEM_WIDTH] = b_lane;

                    for (bit_a = 0; bit_a < ELEM_WIDTH; bit_a = bit_a + 1) begin
                        for (bit_b = 0; bit_b < ELEM_WIDTH; bit_b = bit_b + 1) begin
                            pp_o[((blk*LANES_PER_BLOCK + lane)*ELEM_WIDTH*ELEM_WIDTH) + (bit_a*ELEM_WIDTH) + bit_b] =
                                a_lane[bit_a] & b_lane[bit_b];
                        end
                    end
                end
            end
        end else if (sparse_mode_sel_i == 3'b010) begin
            for (blk = 0; blk < BLOCKS; blk = blk + 1) begin
                idx0 = meta_i[blk*4 +: 2];
                a_lane = a_comp_i[(blk*LANES_PER_BLOCK + 0)*ELEM_WIDTH +: ELEM_WIDTH];
                b_lane = pick_b_elem(blk, idx0);

                lane_valid_o[blk*LANES_PER_BLOCK + 0] = 1'b1;
                a_sel_o[(blk*LANES_PER_BLOCK + 0)*ELEM_WIDTH +: ELEM_WIDTH] = a_lane;
                b_sel_o[(blk*LANES_PER_BLOCK + 0)*ELEM_WIDTH +: ELEM_WIDTH] = b_lane;

                for (bit_a = 0; bit_a < ELEM_WIDTH; bit_a = bit_a + 1) begin
                    for (bit_b = 0; bit_b < ELEM_WIDTH; bit_b = bit_b + 1) begin
                        pp_o[((blk*LANES_PER_BLOCK + 0)*ELEM_WIDTH*ELEM_WIDTH) + (bit_a*ELEM_WIDTH) + bit_b] =
                            a_lane[bit_a] & b_lane[bit_b];
                    end
                end
            end
        end
    end
endmodule