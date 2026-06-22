
module fpadd4 #(
    parameter IN_EXP_WIDTH  = 5,
    parameter IN_MAN_WIDTH  = 3,
    parameter OUT_EXP_WIDTH = 8,
    parameter OUT_MAN_WIDTH = 13,
    parameter LATENCY = 4
)(
    input                              clk,
    input                              rst_n,
    input  [(IN_EXP_WIDTH+IN_MAN_WIDTH+1)*4-1:0] in_i,
    input  [2:0]                        rm_i,

    output reg [OUT_EXP_WIDTH+OUT_MAN_WIDTH:0] result_o,
    output reg [4:0]                    fflags_o
);

localparam IN_FPLEN  = IN_EXP_WIDTH  + IN_MAN_WIDTH  + 1;
localparam OUT_FPLEN = OUT_EXP_WIDTH + OUT_MAN_WIDTH + 1;

localparam SUM_GROWTH = 2;  // 2 bits growth for 4 numbers addition
localparam GRS_WIDTH  = 3;

/* accumulator width */

localparam ACC_MAN_WIDTH =
      1  // sign
    + SUM_GROWTH
    + 1  // hidden bit
    + OUT_MAN_WIDTH
    + GRS_WIDTH;

/*exponent bias */

localparam IN_EXP_BIAS  = (1<<(IN_EXP_WIDTH-1)) - 1;
localparam OUT_EXP_BIAS = (1<<(OUT_EXP_WIDTH-1)) - 1;
localparam BIAS_DIFF    = OUT_EXP_BIAS - IN_EXP_BIAS;
///////////////////////////////////////////////////////////////
//// Stage0 : input register
///////////////////////////////////////////////////////////////
reg [(IN_EXP_WIDTH+IN_MAN_WIDTH+1)*4-1:0] in_s1;
reg [2:0] rm_s1;
always @(posedge clk,negedge rst_n) begin
    if(~rst_n) begin
        in_s1 <= 'd0;
        rm_s1 <= 'd0; end
    else begin
        in_s1 <= in_i;
        rm_s1 <= rm_i;
    end
end
///////////////////////////////////////////////////////////////
//// Stage1 : unpack + align
///////////////////////////////////////////////////////////////

wire [3:0] sign_s1;
wire [IN_EXP_WIDTH-1:0] exp_s1 [3:0];
wire [IN_MAN_WIDTH:0]   man_s1 [3:0];

wire [3:0] is_zero_s1,is_inf_s1,is_nan_s1;

genvar i;

generate
for(i=0;i<4;i=i+1) begin:UNPACK

wire [IN_FPLEN-1:0] fp;
assign fp = in_s1[i*IN_FPLEN +: IN_FPLEN];

assign sign_s1[i] = fp[IN_FPLEN-1];
assign exp_s1[i]  = fp[IN_FPLEN-2 -: IN_EXP_WIDTH];

assign is_zero_s1[i] = (exp_s1[i]==0);

assign is_inf_s1[i] =
    (exp_s1[i]=={IN_EXP_WIDTH{1'b1}}) &&
    (fp[IN_MAN_WIDTH-1:0]==0);

assign is_nan_s1[i] =
    (exp_s1[i]=={IN_EXP_WIDTH{1'b1}}) &&
    (fp[IN_MAN_WIDTH-1:0]!=0);

assign man_s1[i] =
    is_zero_s1[i] ?
    {1'b0,fp[IN_MAN_WIDTH-1:0]} :
    {1'b1,fp[IN_MAN_WIDTH-1:0]};

end
endgenerate

/* invalid */
wire invalid_s1;
assign invalid_s1 = (|is_nan_s1);

/* max exponent */
wire [IN_EXP_WIDTH-1:0] exp_max_01;
wire [IN_EXP_WIDTH-1:0] exp_max_23;
wire [IN_EXP_WIDTH-1:0] exp_max_s1;
assign exp_max_01 = exp_s1[0]>exp_s1[1] ? exp_s1[0] : exp_s1[1];
assign exp_max_23 = exp_s1[2]>exp_s1[3] ? exp_s1[2] : exp_s1[3];
assign exp_max_s1 = exp_max_01>exp_max_23 ? exp_max_01 : exp_max_23;

/* alignment */
wire signed [ACC_MAN_WIDTH-1:0] aligned_man_s1 [3:0];
generate
for(i=0;i<4;i=i+1) begin:ALIGN

wire [IN_EXP_WIDTH-1:0] exp_diff;
assign exp_diff = exp_max_s1 - exp_s1[i];

wire [ACC_MAN_WIDTH-1:0] man_ext;

assign man_ext = {
    {1'b0}, // sign will be applied after shift
    {(SUM_GROWTH){1'b0}},
    man_s1[i],
    {OUT_MAN_WIDTH-IN_MAN_WIDTH{1'b0}},
    {GRS_WIDTH{1'b0}}

};

wire [ACC_MAN_WIDTH-1:0] shifted;
wire sticky;

shift_right_jam #(
    .LEN (ACC_MAN_WIDTH),
    .EXP (IN_EXP_WIDTH)
) u_shift (
    .in     (man_ext),
    .shamt  (exp_diff),
    .out    (shifted),
    .sticky (sticky)
);

wire [ACC_MAN_WIDTH-1:0] jam;

assign jam =
    {shifted[ACC_MAN_WIDTH-1:1],
     shifted[0] | sticky};

assign aligned_man_s1[i] =
    sign_s1[i] ?
    -$signed(jam) : $signed(jam);

end
endgenerate

/* pipeline */
reg signed [ACC_MAN_WIDTH-1:0] man_s2 [3:0];
reg [OUT_EXP_WIDTH-1:0] exp_s2;
reg invalid_s2;
reg [2:0] rm_s2;

integer k;

always @(posedge clk or negedge rst_n)
begin
if(!rst_n) begin
exp_s2<=0;
invalid_s2<=0;
rm_s2<=0;

for(k=0;k<4;k=k+1) man_s2[k]<=0;
end
else begin
exp_s2<=exp_max_s1;
invalid_s2<=invalid_s1;
rm_s2<=rm_s1;
for(k=0;k<4;k=k+1) man_s2[k]<=aligned_man_s1[k];
end
end

///////////////////////////////////////////////////////////////
//// Stage2 : CSA + CPA
///////////////////////////////////////////////////////////////

wire [ACC_MAN_WIDTH-1:0] csa1_sum,csa1_carry;
wire [ACC_MAN_WIDTH-1:0] csa2_sum,csa2_carry;

csa3_2 #(.WIDTH(ACC_MAN_WIDTH)) u_csa1(
.a(man_s2[0]),
.b(man_s2[1]),
.c(man_s2[2]),
.sum(csa1_sum),
.carry(csa1_carry)
);

csa3_2 #(.WIDTH(ACC_MAN_WIDTH)) u_csa2(
.a(csa1_sum),
.b(csa1_carry<<1),
.c(man_s2[3]),
.sum(csa2_sum),
.carry(csa2_carry)
);

/* CPA */
wire signed [ACC_MAN_WIDTH-1:0] final_sum_ext;

assign final_sum_ext = csa2_sum + (csa2_carry<<1);

wire final_sign;
assign final_sign = final_sum_ext[ACC_MAN_WIDTH-1];

wire [ACC_MAN_WIDTH-2:0] final_abs;
wire [ACC_MAN_WIDTH-1:0] final_sum_inv;
assign final_sum_inv = ~final_sum_ext + 1'b1;  
assign final_abs =
    final_sign ?
    final_sum_inv[ACC_MAN_WIDTH-2:0] :
    final_sum_ext[ACC_MAN_WIDTH-2:0];

/* LZC directly on CPA result */
wire [$clog2(ACC_MAN_WIDTH):0] lzc_real;
wire lzc_empty;

lzc #(
.WIDTH(ACC_MAN_WIDTH-1),
.MODE(1'b1)
) u_lzc(
.in_i(final_abs),
.cnt_o(lzc_real),
.empty_o(lzc_empty)
);

/* pipeline */
reg [ACC_MAN_WIDTH -2:0] abs_s3;
reg sign_s3;
reg [OUT_EXP_WIDTH-1:0] exp_s3;
reg [$clog2(ACC_MAN_WIDTH):0] lzc_s3;
reg invalid_s3;
reg [2:0] rm_s3;
reg zero_flag_s3;

always @(posedge clk or negedge rst_n)
begin
if(!rst_n) begin
abs_s3<=0;
sign_s3<=0;
exp_s3<=0;
lzc_s3<=0;
invalid_s3<=0;
rm_s3<=0;
zero_flag_s3<=0;
end
else begin
abs_s3<=final_abs;
sign_s3<=final_sign;
exp_s3<=exp_s2;
lzc_s3<=lzc_empty ? 0 : lzc_real;
invalid_s3<=invalid_s2;
rm_s3<=rm_s2;
zero_flag_s3<= lzc_empty;
end
end

///////////////////////////////////////////////////////////////
//// Stage3 : normalize + rounding
///////////////////////////////////////////////////////////////

wire [ACC_MAN_WIDTH-2:0] norm_man;  //without hidden bit

assign norm_man = abs_s3 << lzc_s3;

/* exponent */
wire signed [OUT_EXP_WIDTH:0] norm_exp;

assign norm_exp =
    $signed({1'b0,exp_s3})
    - $signed(lzc_s3)
    + SUM_GROWTH;

/* bias convert */
wire [OUT_EXP_WIDTH:0] exp_bias_conv;

assign exp_bias_conv =
    norm_exp + BIAS_DIFF;

/* rounding bits */
wire rounding;
wire sticky;

assign rounding = norm_man[GRS_WIDTH+2-1]; //SUM_GROWTH+GRS_WIDTH

assign sticky =|norm_man[GRS_WIDTH+2-2:2];

wire [OUT_MAN_WIDTH:0] round_in; // with hidden bit

assign round_in = norm_man[ACC_MAN_WIDTH-2 -: (OUT_MAN_WIDTH+1)];

wire [OUT_MAN_WIDTH:0] round_out;
wire inexact;
wire cout;
wire r_up;

rounding #(.WIDTH(OUT_MAN_WIDTH+1)) u_round(
.in(round_in),
.sign(sign_s3),
.roundin(rounding),
.stickyin(sticky),
.rm(rm_i),
.out(round_out),
.inexact(inexact),
.cout(cout),
.r_up(r_up)
);

/* exponent final */
wire [OUT_EXP_WIDTH-1:0] exp_final;
assign exp_final = zero_flag_s3 ? 0 :  exp_bias_conv + cout;

/* flags */
wire overflow;
wire underflow;

assign overflow =
    (exp_final == {OUT_EXP_WIDTH{1'b1}});

assign underflow =
    (exp_final == 0) && inexact;

wire nx = inexact;
wire dz = 1'b0;
wire nv = invalid_s3;

always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        result_o <= 0;
        fflags_o <= 0; end
    else begin
        fflags_o = {nv,dz,overflow,underflow,nx};
        if(nv)
            result_o = {1'b0,{OUT_EXP_WIDTH{1'b1}},1'b1,{OUT_MAN_WIDTH-1{1'b0}}};
        else if(overflow)
            result_o = {sign_s3,{OUT_EXP_WIDTH{1'b1}},{OUT_MAN_WIDTH{1'b0}}};
        else
            result_o = {sign_s3,exp_final,round_out[OUT_MAN_WIDTH-1:0]};
        end
end

endmodule
