
//这是一个简化版本，测试通过

// module fp_to_fp22 #(
//     parameter MATRIX_BUS_WIDTH = 512,
//     parameter OUT_BUS_WIDTH = 1408
// )(
//     input                       clk,
//     input                       rst_n,

//     input  [MATRIX_BUS_WIDTH-1:0] in_i,
//     input  [1:0]                  dtype_i,   //,11:fp32,10:fp16,01:e4m3,00:e5m2

//     input                         in_valid_i,
//     input                         out_ready_i,

//     output [OUT_BUS_WIDTH-1:0]    out_o,
//     output                        out_valid_o,
//     output                        in_ready_o
// );

//     // ============================================================
//     // 参数定义
//     // ============================================================
//     localparam FP22_WIDTH = 22;
//     localparam OUT_NUM    = 64;

//     // ============================================================
//     // ready/valid（纯组合直通）
//     // ============================================================
//     assign out_valid_o = in_valid_i;
//     assign in_ready_o  = out_ready_i;

//     // ============================================================
//     // 输出寄存（组合）
//     // ============================================================
// reg [1407:0] out_r;
// assign out_o = out_r;

// integer i;

// // 统一声明（关键！）
// reg [7:0]  d8;
// reg [15:0] d16;
// reg [31:0] d32;

// reg sign;
// reg [7:0] exp_out;
// reg [12:0] man_out;

// reg [3:0] exp4;
// reg [4:0] exp5;
// reg [7:0] exp8;

// reg [2:0] man3;
// reg [1:0] man2;
// reg [9:0] man10;
// reg [22:0] man23;

// always @(*) begin
//     out_r = 0;

//     case(dtype_i)

//     // ========================================================
//     // FP8 E4M3
//     // ========================================================
//     2'b01: begin
//         for(i = 0; i < 64; i = i + 1) begin

//             d8   = in_i[i*8 +: 8];
//             sign = d8[7];
//             exp4 = d8[6:3];
//             man3 = d8[2:0];

//             exp_out = (exp4 == 0) ? 0 : (exp4 - 7 + 127);
//             man_out = {man3, 10'b0};

//             out_r[i*22 +: 22] = {sign, exp_out, man_out};
//         end
//     end

//     // ========================================================
//     // FP8 E5M2
//     // ========================================================
//     2'b00: begin
//         for(i = 0; i < 64; i = i + 1) begin

//             d8   = in_i[i*8 +: 8];
//             sign = d8[7];
//             exp5 = d8[6:2];
//             man2 = d8[1:0];

//             exp_out = (exp5 == 0) ? 0 : (exp5 - 15 + 127);
//             man_out = {man2, 11'b0};

//             out_r[i*22 +: 22] = {sign, exp_out, man_out};
//         end
//     end

//     // ========================================================
//     // FP16
//     // ========================================================
//     2'b10: begin
//         for(i = 0; i < 32; i = i + 1) begin

//             d16  = in_i[i*16 +: 16];
//             sign = d16[15];
//             exp5 = d16[14:10];
//             man10 = d16[9:0];

//             exp_out = (exp5 == 0) ? 0 : (exp5 - 15 + 127);
//             man_out = {man10, 3'b0};

//             out_r[i*22 +: 22] = {sign, exp_out, man_out};
//         end
//     end

//     // ========================================================
//     // FP32
//     // ========================================================
//     2'b11: begin
//         for(i = 0; i < 16; i = i + 1) begin

//             d32  = in_i[i*32 +: 32];
//             sign = d32[31];
//             exp8 = d32[30:23];
//             man23 = d32[22:0];

//             exp_out = exp8;
//             man_out = man23[22:10];

//             out_r[i*22 +: 22] = {sign, exp_out, man_out};
//         end
//     end

//     default: out_r = 0;

//     endcase
// end

// endmodule


module fp_to_fp22 #(
    parameter MATRIX_BUS_WIDTH = 512,
    parameter OUT_BUS_WIDTH    = 1408
) (
    input                           clk,
    input                           rst_n,

    input  [MATRIX_BUS_WIDTH-1:0]   in_i,
    input  [1:0]                    dtype_i,  // 11:fp32, 10:fp16, 01:e4m3, 00:e5m2

    input                           in_valid_i,
    input                           out_ready_i,

    output [OUT_BUS_WIDTH-1:0]      out_o,
    output                          out_valid_o,
    output                          in_ready_o
);

    // ============================================================
    // 参数定义
    // ============================================================
    localparam FP22_WIDTH = 22;
    localparam OUT_NUM    = 64;

    // FP22 constants
    localparam FP22_EXP_BIAS = 8'd127;
    localparam FP22_EXP_MAX  = 8'd254;  // max normal exponent (bias 127, exp 1..254)
    localparam FP22_EXP_SPECIAL = 8'd255;  // Inf/NaN exponent

    // ============================================================
    // ready/valid（纯组合直通）
    // ============================================================
    assign out_valid_o = in_valid_i;
    assign in_ready_o  = out_ready_i;

    // ============================================================
    // 输出寄存（组合）
    // ============================================================
    reg [1407:0] out_r;
    assign out_o = out_r;

    // ============================================================
    // 内部信号声明
    // ============================================================
    integer i;

    // 输入解析
    reg [7:0]   d8;
    reg [15:0]  d16;
    reg [31:0]  d32;

    // 输入分解
    reg         sign;
    reg [4:0]   exp5;     // E5M2 / FP16 exponent
    reg [3:0]   exp4;     // E4M3 exponent
    reg [7:0]   exp8;     // FP32 exponent
    reg [2:0]   man3;     // E4M3 mantissa
    reg [1:0]   man2;     // E5M2 mantissa
    reg [9:0]   man10;    // FP16 mantissa
    reg [22:0]  man23;    // FP32 mantissa

    // FP22 输出
    reg         sign_out;
    reg [7:0]   exp_out;
    reg [12:0]  man_out;

    // RNE rounding wires (FP32 path)
    wire        rnd_bit;   // man23[9]
    wire        sticky;    // |man23[8:0]
    wire        rnd_inc;   // RNE: R & (S | LSB)

    // FP16 subnormal: 前导零计数 (在 always block 内计算, 每个元素独立)
    // lzc_cnt = 前导零个数 k (0..9), 即 man10 最高位 1 的位置 = 9 - k
    // exp_out = 103 + (9 - k) = 112 - k
    // man_out = {man10 << (k+1), 3'b0}
    reg [3:0] lzc_cnt;

    // ============================================================
    // RNE 舍入信号 (FP32 path)
    // ============================================================
    assign sticky  = |man23[8:0];   // OR of all sticky bits
    assign rnd_bit = man23[9];     // round bit (first dropped bit)
    // RNE: round up if (R=1 and S=1) or (R=1 and S=0 and LSB=1)
    assign rnd_inc = rnd_bit & (sticky | man23[10]);

    // ============================================================
    // 主转换逻辑
    // ============================================================
    always @(*) begin
        out_r = 0;

        case (dtype_i)

        // ========================================================
        // FP8 E4M3
        // ========================================================
        2'b01: begin
            for (i = 0; i < 64; i = i + 1) begin
                d8   = in_i[i*8 +: 8];
                sign = d8[7];
                exp4 = d8[6:3];
                man3 = d8[2:0];

                // --- 特殊值: NaN ---
                if (exp4 == 4'b1111 && man3 == 3'b111) begin
                    sign_out = sign;
                    exp_out  = FP22_EXP_SPECIAL;  // 255
                    man_out  = {1'b1, 12'b0};     // qNaN: quiet bit set
                end
                // --- 次规范化数 (exp4 == 0, man3 != 0) ---
                else if (exp4 == 4'b0000 && man3 != 3'b000) begin
                    case (man3)
                        3'b001: begin exp_out = 8'd118; man_out = 13'b0; end
                        3'b010: begin exp_out = 8'd119; man_out = 13'b0; end
                        3'b011: begin exp_out = 8'd119; man_out = {1'b1, 12'b0}; end
                        3'b100: begin exp_out = 8'd120; man_out = 13'b0; end
                        3'b101: begin exp_out = 8'd120; man_out = {1'b0, 1'b1, 11'b0}; end
                        3'b110: begin exp_out = 8'd120; man_out = {1'b1, 1'b0, 10'b0}; end
                        3'b111: begin exp_out = 8'd120; man_out = {1'b1, 1'b1, 10'b0}; end
                        default: begin exp_out = 8'd0; man_out = 13'b0; end
                    endcase
                    sign_out = sign;
                end
                // --- 规格数 ---
                else begin
                    sign_out = sign;
                    exp_out  = (exp4 == 0) ? 8'd0 : (exp4 - 8'd7 + FP22_EXP_BIAS);
                    man_out  = {man3, 10'b0};  // 零扩展到 13 位
                end

                out_r[i*22 +: 22] = {sign_out, exp_out, man_out};
            end
        end

        // ========================================================
        // FP8 E5M2
        // ========================================================
        2'b00: begin
            for (i = 0; i < 64; i = i + 1) begin
                d8   = in_i[i*8 +: 8];
                sign = d8[7];
                exp5 = d8[6:2];
                man2 = d8[1:0];

                // --- 特殊值: Infinity ---
                if (exp5 == 5'b11111 && man2 == 2'b00) begin
                    sign_out = sign;
                    exp_out  = FP22_EXP_SPECIAL;  // 255
                    man_out  = 13'b0;
                end
                // --- 特殊值: NaN ---
                else if (exp5 == 5'b11111 && man2 != 2'b00) begin
                    sign_out = sign;
                    exp_out  = FP22_EXP_SPECIAL;  // 255
                    man_out  = {1'b1, 12'b0};     // qNaN
                end
                // --- 次规范化数 (exp5 == 0, man2 != 0) ---
                else if (exp5 == 5'b00000 && man2 != 2'b00) begin
                    sign_out = sign;
                    case (man2)
                        2'b01: begin exp_out = 8'd111; man_out = 13'b0; end
                        2'b10: begin exp_out = 8'd112; man_out = 13'b0; end
                        2'b11: begin exp_out = 8'd112; man_out = {1'b1, 12'b0}; end
                        default: begin exp_out = 8'd0; man_out = 13'b0; end
                    endcase
                end
                // --- 规格数 ---
                else begin
                    sign_out = sign;
                    exp_out  = (exp5 == 0) ? 8'd0 : (exp5 - 5'd15 + FP22_EXP_BIAS);
                    man_out  = {man2, 11'b0};  // 零扩展到 13 位
                end

                out_r[i*22 +: 22] = {sign_out, exp_out, man_out};
            end
        end

        // ========================================================
        // FP16
        // ========================================================
        2'b10: begin
            for (i = 0; i < 32; i = i + 1) begin
                d16  = in_i[i*16 +: 16];
                sign = d16[15];
                exp5 = d16[14:10];
                man10 = d16[9:0];

                // --- 特殊值: Infinity ---
                if (exp5 == 5'b11111 && man10 == 10'b0) begin
                    sign_out = sign;
                    exp_out  = FP22_EXP_SPECIAL;  // 255
                    man_out  = 13'b0;
                end
                // --- 特殊值: NaN ---
                else if (exp5 == 5'b11111 && man10 != 10'b0) begin
                    sign_out = sign;
                    exp_out  = FP22_EXP_SPECIAL;  // 255
                    man_out  = {1'b1, 12'b0};       // qNaN
                end
                // --- 次规范化数 (exp5 == 0, man10 != 0) ---
                else if (exp5 == 5'b00000 && man10 != 10'b0) begin
                    sign_out = sign;
                    // 计算前导零个数: priority encoder 找最高位 1 的位置
                    casez (man10)
                        10'b1?????????: lzc_cnt = 4'd0;   // MSB is 1
                        10'b01????????: lzc_cnt = 4'd1;
                        10'b001????????: lzc_cnt = 4'd2;
                        10'b0001???????: lzc_cnt = 4'd3;
                        10'b00001??????: lzc_cnt = 4'd4;
                        10'b000001?????: lzc_cnt = 4'd5;
                        10'b0000001????: lzc_cnt = 4'd6;
                        10'b00000001???: lzc_cnt = 4'd7;
                        10'b000000001??: lzc_cnt = 4'd8;
                        10'b0000000001?: lzc_cnt = 4'd9;
                        default:        lzc_cnt = 4'd0;    // shouldn't happen (man10 != 0)
                    endcase
                    // k = lzc_cnt (前导零), 最高位1位置 = 9-k
                    // exp_out = 103 + (9-k) = 112 - k
                    exp_out = 8'd112 - {3'b0, lzc_cnt};
                    man_out = {man10 << (lzc_cnt + 4'd1), 3'b0};
                end
                // --- 规格数 ---
                else begin
                    sign_out = sign;
                    exp_out  = (exp5 == 0) ? 8'd0 : (exp5 - 5'd15 + FP22_EXP_BIAS);
                    man_out  = {man10, 3'b0};  // 零扩展到 13 位
                end

                out_r[i*22 +: 22] = {sign_out, exp_out, man_out};
            end
        end

        // ========================================================
        // FP32
        // ========================================================
        2'b11: begin
            for (i = 0; i < 16; i = i + 1) begin
                d32   = in_i[i*32 +: 32];
                sign  = d32[31];
                exp8  = d32[30:23];
                man23 = d32[22:0];

                // --- 特殊值: Infinity ---
                if (exp8 == 8'hFF && man23 == 23'b0) begin
                    sign_out = sign;
                    exp_out  = FP22_EXP_SPECIAL;  // 255
                    man_out  = 13'b0;
                end
                // --- 特殊值: NaN ---
                else if (exp8 == 8'hFF && man23 != 23'b0) begin
                    sign_out = sign;
                    exp_out  = FP22_EXP_SPECIAL;    // 255
                    man_out  = {1'b1, 12'b0};       // qNaN (quiet bit set)
                end
                // --- 次规范化数 (exp8 == 0, man23 != 0) ---
                else if (exp8 == 8'd0 && man23 != 23'b0) begin
                    sign_out = sign;
                    exp_out  = 8'd0;
                    // RNE 舍入: man23[22:10] + rnd_inc
                    // 进位时 mantissa 清零, exp_out 保持 0 (subnormal 溢出到最小规格数)
                    if (rnd_inc) begin
                        if (man23[22:10] == 13'h1FFF) begin
                            // 溢出: 回到最小规格数
                            man_out  = 13'b0;
                            // subnormal 溢出 → 最小规格数 exp=1, man=0
                            exp_out  = 8'd1;
                        end else begin
                            man_out = man23[22:10] + 13'd1;
                        end
                    end else begin
                        man_out = man23[22:10];
                    end
                end
                // --- 规格数 ---
                else begin
                    sign_out = sign;
                    exp_out  = exp8;
                    // RNE 舍入: man23[22:10] + rnd_inc
                    if (rnd_inc) begin
                        if (man23[22:10] == 13'h1FFF) begin
                            // 溢出到 Inf
                            exp_out  = FP22_EXP_SPECIAL;  // 255
                            man_out  = 13'b0;
                        end else begin
                            man_out = man23[22:10] + 13'd1;
                        end
                    end else begin
                        man_out = man23[22:10];
                    end
                end

                out_r[i*22 +: 22] = {sign_out, exp_out, man_out};
            end
        end

        default: out_r = 0;

        endcase
    end

endmodule