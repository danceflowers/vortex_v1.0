// `timescale 1ns/1ns
// `include "define.v"

// module fp_to_fp9 #(
//     parameter MATRIX_BUS_WIDTH = `MATRIX_BUS_WIDTH,
//     parameter LATENCY = 1
// )(
//     input                       clk,
//     input                       rst_n,

//     input  [MATRIX_BUS_WIDTH-1:0] in_i,
//     input  [1:0]                  dtype_i,   //10:fp16,01:fp8(e4m3),00:fp8(e5m2)

//     input                         in_valid_i,
//     input                         out_ready_i,

//     output [MATRIX_BUS_WIDTH*9/8-1:0] out_o,
//     output                            out_valid_o,
//     output                            in_ready_o
// );

// localparam FP8_NUM  = MATRIX_BUS_WIDTH / 8;
// localparam FP16_NUM = MATRIX_BUS_WIDTH / 16;
// wire fp16;
// wire fp8_e4;
// wire fp8_e5;

// assign fp16 = dtype_i==2'b10;
// assign fp8_e4 = dtype_i==2'b01;
// assign fp8_e5 = dtype_i==2'b00;
// integer i;

// //--------------------------------------------------
// // conversion combinational
// //--------------------------------------------------
// reg [MATRIX_BUS_WIDTH*9/8-1:0] convert_bus;

// always @(*) begin
//     convert_bus = 0;

//     if(fp16) begin  // FP16 -> FP9
//         for(i=0;i<FP16_NUM;i=i+1) begin
//             convert_bus[i*9 +: 9] =
//                 fp16_to_fp9(in_i[i*16 +: 16]);
//         end
//     end
//     else if(fp8_e4) begin // FP8 E4M3 -> FP9
//         for(i=0;i<FP8_NUM;i=i+1) begin
//             convert_bus[i*9 +: 9] =
//                 fp8e4m3_to_fp9(in_i[i*8 +: 8]);
//         end
//     end
//     else /* if(fp8_e5) */ begin // FP8 E5M2 -> FP9
//         for(i=0;i<FP8_NUM;i=i+1) begin
//             convert_bus[i*9 +: 9] =
//                 fp8e5m2_to_fp9(in_i[i*8 +: 8]);
//         end
//     end 
// end

// //--------------------------------------------------
// // pipeline registers with valid/ready
// //--------------------------------------------------
// reg [MATRIX_BUS_WIDTH*9/8-1:0] out_pipe;
// reg out_valid_pipe;

// assign in_ready_o = out_ready_i;  // simple backpressure

// always @(posedge clk or negedge rst_n) begin
//     if(!rst_n) begin
//         out_pipe       <= 0;
//         out_valid_pipe <= 0;

//     end
//     else begin
//         if(in_valid_i && out_ready_i) begin
//             out_pipe       <= convert_bus;
//             out_valid_pipe <= 1'b1;
//         end
//         else if(out_ready_i) begin
//             out_valid_pipe <= 1'b0;
//         end
//     end
// end




// assign    out_o       = out_pipe;
// assign    out_valid_o = out_valid_pipe;

// //--------------------------------------------------
// // FP16 -> FP9
// //--------------------------------------------------
// function [8:0] fp16_to_fp9;
//     input [15:0] in;
//     reg sign;
//     reg [4:0] exp;
//     reg [9:0] man;
// begin
//     sign = in[15];
//     exp  = in[14:10];
//     man  = in[9:0];
//     fp16_to_fp9 = {sign, exp, man[9:7]};
// end
// endfunction

// //--------------------------------------------------
// // FP8(E4M3) -> FP9
// //--------------------------------------------------
// function [8:0] fp8e4m3_to_fp9;
//     input [7:0] in;
//     reg sign;
//     reg [3:0] exp;
//     reg [2:0] man;
//     reg [4:0] exp_adjusted;
// begin
//     sign = in[7];
//     exp  = in[6:3];
//     man  = in[2:0];
//     exp_adjusted = (exp==0) ? 0 : ({1'd0,exp} + 5'd8); // bias difference
//     fp8e4m3_to_fp9 = {sign,exp_adjusted,man};
// end
// endfunction

// //--------------------------------------------------
// // FP8(E5M2) -> FP9
// //--------------------------------------------------
// function [8:0] fp8e5m2_to_fp9;
//     input [7:0] in;
//     reg sign;
//     reg [4:0] exp;
//     reg [1:0] man;
// begin
//     sign = in[7];
//     exp  = in[6:2];
//     man  = in[1:0];
//     fp8e5m2_to_fp9 = {sign,exp,man,1'b0};
// end
// endfunction

// endmodule


`timescale 1ns/1ns
`include "define.v"
module fp_to_fp9 #(
    parameter MATRIX_BUS_WIDTH = `MATRIX_BUS_WIDTH,
    parameter LATENCY = 1
)(
    input                       clk,
    input                       rst_n,
    input  [MATRIX_BUS_WIDTH-1:0] in_i,
    input  [1:0]                  dtype_i,   //10:fp16,01:fp8(e4m3),00:fp8(e5m2)
    input                         in_valid_i,
    input                         out_ready_i,
    output [MATRIX_BUS_WIDTH*9/8-1:0] out_o,
    output                            out_valid_o,
    output                            in_ready_o
);
localparam FP8_NUM  = MATRIX_BUS_WIDTH / 8;
localparam FP16_NUM = MATRIX_BUS_WIDTH / 16;
wire fp16;
wire fp8_e4;
wire fp8_e5;
assign fp16   = dtype_i == 2'b10;
assign fp8_e4 = dtype_i == 2'b01;
assign fp8_e5 = dtype_i == 2'b00;
integer i;
//--------------------------------------------------
// conversion combinational
//--------------------------------------------------
reg [MATRIX_BUS_WIDTH*9/8-1:0] convert_bus;
always @(*) begin
    convert_bus = 0;
    if(fp16) begin  // FP16 -> FP9
        for(i=0;i<FP16_NUM;i=i+1) begin
            convert_bus[i*9 +: 9] =
                fp16_to_fp9(in_i[i*16 +: 16]);
        end
    end
    else if(fp8_e4) begin // FP8 E4M3 -> FP9
        for(i=0;i<FP8_NUM;i=i+1) begin
            convert_bus[i*9 +: 9] =
                fp8e4m3_to_fp9(in_i[i*8 +: 8]);
        end
    end
    else /* if(fp8_e5) */ begin // FP8 E5M2 -> FP9
        for(i=0;i<FP8_NUM;i=i+1) begin
            convert_bus[i*9 +: 9] =
                fp8e5m2_to_fp9(in_i[i*8 +: 8]);
        end
    end 
end
//--------------------------------------------------
// pipeline registers with valid/ready
//--------------------------------------------------
reg [MATRIX_BUS_WIDTH*9/8-1:0] out_pipe;
reg out_valid_pipe;
assign in_ready_o = out_ready_i;  // simple backpressure
always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        out_pipe       <= 0;
        out_valid_pipe <= 0;
    end
    else begin
        if(in_valid_i && out_ready_i) begin
            out_pipe       <= convert_bus;
            out_valid_pipe <= 1'b1;
        end
        else if(out_ready_i) begin
            out_valid_pipe <= 1'b0;
        end
    end
end
assign out_o       = out_pipe;
assign out_valid_o = out_valid_pipe;

//--------------------------------------------------
// FP16 -> FP9
// FP16 : 1s + 5e(bias=15) + 10m
// FP9  : 1s + 5e(bias=15) +  3m   (same bias, truncate/round mantissa)
// Rounding: Round-to-Nearest-Even on discarded mantissa bits [6:0]
//--------------------------------------------------
function [8:0] fp16_to_fp9;
    input [15:0] in;
    reg        sign;
    reg [4:0]  exp;
    reg [9:0]  man;
    reg [2:0]  man3;      // top 3 mantissa bits
    reg [6:0]  guard;     // discarded bits [6:0]
    reg        round_up;
    reg [2:0]  man3_r;    // rounded mantissa
    reg [4:0]  exp_r;     // rounded exponent
begin
    sign  = in[15];
    exp   = in[14:10];
    man   = in[9:0];

    // ---- special values ----
    if (exp == 5'h1F) begin
        if (man == 10'h0)
            // Infinity -> FP9 Infinity
            fp16_to_fp9 = {sign, 5'h1F, 3'h0};
        else
            // NaN -> FP9 NaN (canonical: mantissa = 001)
            fp16_to_fp9 = {sign, 5'h1F, 3'h1};
    end
    else if (exp == 5'h00) begin
        if (man == 10'h0) begin
            // +/- Zero
            fp16_to_fp9 = {sign, 5'h00, 3'h0};
        end
        else begin
            // FP16 subnormal: value = 2^(-14) * 0.man10
            // Represent as FP9 subnormal: 2^(-14) * 0.man3
            // man3 = man[9:7], guard = man[6:0]; apply RNE
            man3  = man[9:7];
            guard = man[6:0];

            round_up = (guard > 7'h40) ||
                       ((guard == 7'h40) && man3[0]);

            man3_r = man3 + {2'h0, round_up};

            if (man3_r == 3'h0 && round_up) begin
                // Mantissa wrapped to 0 -> promote to smallest normal
                fp16_to_fp9 = {sign, 5'h01, 3'h0};
            end
            else begin
                fp16_to_fp9 = {sign, 5'h00, man3_r};
            end
        end
    end
    else begin
        // Normal number: RNE on man[6:0]
        man3  = man[9:7];
        guard = man[6:0];

        round_up = (guard > 7'h40) ||
                   ((guard == 7'h40) && man3[0]);

        man3_r = man3 + {2'h0, round_up};
        exp_r  = exp;

        if (man3_r == 3'h0 && round_up) begin
            // Mantissa overflow -> increment exponent
            exp_r = exp + 5'h1;
            if (exp_r == 5'h1F) begin
                // Overflow to Infinity
                fp16_to_fp9 = {sign, 5'h1F, 3'h0};
            end
            else begin
                fp16_to_fp9 = {sign, exp_r, 3'h0};
            end
        end
        else begin
            fp16_to_fp9 = {sign, exp_r, man3_r};
        end
    end
end
endfunction

//--------------------------------------------------
// FP8(E4M3) -> FP9
// E4M3 : 1s + 4e(bias=7)  + 3m   NaN = S_1111_111 (no Inf)
// FP9  : 1s + 5e(bias=15) + 3m
// Normal  : new_exp = old_exp - 7 + 15 = old_exp + 8  (mantissa unchanged)
// Subnorm : normalize; find leading 1, adjust exponent accordingly
// NaN     : exp=1111, man!=0 -> FP9 NaN
// Zero    : exp=0,    man=0   -> FP9 Zero
//--------------------------------------------------
function [8:0] fp8e4m3_to_fp9;
    input [7:0] in;
    reg        sign;
    reg [3:0]  exp;
    reg [2:0]  man;
    reg [4:0]  exp9;
    reg [2:0]  man9;
begin
    sign = in[7];
    exp  = in[6:3];
    man  = in[2:0];

    // ---- NaN: exp=1111, man!=0 ----
    if (exp == 4'hF && man != 3'h0) begin
        fp8e4m3_to_fp9 = {sign, 5'h1F, 3'h1};
    end
    // ----infine: exp=1111, man=0 ----
    if (exp == 4'hF && man == 3'h0) begin
        fp8e4m3_to_fp9 = {sign, 5'h1F, 3'h0};
    end
    // ---- Zero ----
    else if (exp == 4'h0 && man == 3'h0) begin
        fp8e4m3_to_fp9 = {sign, 5'h0, 3'h0};
    end
    // ---- Subnormal E4M3: exp=0, man!=0 ----
    // value = 2^(1-7) * 0.man = 2^(-6) * 0.man3
    // Normalize: shift mantissa left until leading 1, adjust exponent.
    // FP9 bias=15; the effective unbiased exp starts at -6 for the implicit 0.
    // Bit positions: man[2]=0.1xx -> shift 1, unbiased_exp = -6-1 = -7 -> exp9 = -7+15 = 8
    //                man[1]=0.01x -> shift 2, unbiased_exp = -8          -> exp9 = 7
    //                man[0]=0.001 -> shift 3, unbiased_exp = -9          -> exp9 = 6
    else if (exp == 4'h0) begin
        if (man[2]) begin
            // 0.1xx -> 1.xx0, exp9 = -7+15 = 8
            exp9 = 5'd8;
            man9 = {man[1:0], 1'b0};
        end
        else if (man[1]) begin
            // 0.01x -> 1.x00, exp9 = -8+15 = 7
            exp9 = 5'd7;
            man9 = {man[0], 2'b00};
        end
        else begin
            // 0.001 -> 1.000, exp9 = -9+15 = 6
            exp9 = 5'd6;
            man9 = 3'h0;
        end
        fp8e4m3_to_fp9 = {sign, exp9, man9};
    end
    // ---- Normal E4M3 ----
    else begin
        exp9 = {1'b0, exp} + 5'd8;   // bias re-encode: -7+15 = +8
        fp8e4m3_to_fp9 = {sign, exp9, man};
    end
end
endfunction

//--------------------------------------------------
// FP8(E5M2) -> FP9
// E5M2 : 1s + 5e(bias=15) + 2m   Inf/NaN like FP16
// FP9  : 1s + 5e(bias=15) + 3m
// Same bias -> exponent passes through unchanged.
// Mantissa: zero-extend LSB (man2 -> {man2, 1'b0})
//--------------------------------------------------
function [8:0] fp8e5m2_to_fp9;
    input [7:0] in;
    reg        sign;
    reg [4:0]  exp;
    reg [1:0]  man;
begin
    sign = in[7];
    exp  = in[6:2];
    man  = in[1:0];

    // ---- special values: exp = 11111 ----
    if (exp == 5'h1F) begin
        if (man == 2'h0)
            // Infinity -> FP9 Infinity
            fp8e5m2_to_fp9 = {sign, 5'h1F, 3'h0};
        else
            // NaN -> FP9 NaN (canonical mantissa = 001)
            fp8e5m2_to_fp9 = {sign, 5'h1F, 3'h1};
    end
    // ---- Zero ----
    else if (exp == 5'h00 && man == 2'h0) begin
        fp8e5m2_to_fp9 = {sign, 5'h0, 3'h0};
    end
    // ---- Subnormal E5M2: exp=0, man!=0 ----
    // Same bias as FP9 -> stays subnormal; zero-extend mantissa
    else if (exp == 5'h00) begin
        // value = 2^(-14) * 0.mm  ->  FP9 subnormal = 2^(-14) * 0.mm0
        fp8e5m2_to_fp9 = {sign, 5'h0, man, 1'b0};
    end
    // ---- Normal ----
    else begin
        // Same bias, same exponent; zero-extend mantissa LSB
        fp8e5m2_to_fp9 = {sign, exp, man, 1'b0};
    end
end
endfunction

endmodule