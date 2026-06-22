module reordering(

input  [575:0] in_data,
input          transpose_en,

output [575:0] out_data

);
genvar r,c;
generate
    for(r=0 ; r<8 ; r=r+1) begin :ROW
    for(c=0 ; c<8 ; c=c+1) begin :COL
        wire[8:0] in_fp9;
        wire[8:0] trans_fp9;

        assign in_fp9 = in_data[(r*8+c)*9 +: 9];
        assign trans_fp9 = in_data[(c*8+r)*9 +: 9];

        assign out_data[(r*8+c)*9 +: 9] = transpose_en ? trans_fp9 : in_fp9;
    end
    end
endgenerate
endmodule