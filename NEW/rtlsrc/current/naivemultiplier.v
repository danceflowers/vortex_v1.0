/*
 * Copyright (c) 2023-2024 C*Core Technology Co.,Ltd,Suzhou.
 * Ventus-RTL is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details. */
// Author: Tan, Zhiyuan
// Description:

`timescale 1ns/1ns

module naivemultiplier #(
        parameter LEN = 32
    )(
        input              clk       ,
        input              rst_n     ,
        input              regenable ,
        input  [LEN-1:0]   a         ,
        input  [LEN-1:0]   b         ,
        output [LEN*2-1:0] result
    );

    reg [LEN-1:0] reg_a ;
    reg [LEN-1:0] reg_b ;

    always@(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            reg_a <= 'd0 ;
            reg_b <= 'd0 ;
        end
        else begin
            if(regenable) begin
                reg_a <= a ;
                reg_b <= b ;
            end
            else begin
                reg_a <= reg_a ;
                reg_b <= reg_b ;
            end
        end
    end

    assign result = reg_a * reg_b ;

endmodule

// `timescale 1ns/1ns
// // =============================================================================
// //  naivemultiplier — Radix-4 Booth 编码乘法器（参数化）
// //
// //  接口与原版完全相同，内部用 Radix-4 Booth 编码替换 "reg_a * reg_b"。
// //
// //  Radix-4 Booth 编码原理：
// //    对乘数 B 每次取3位（重叠1位）：b[2i+1], b[2i], b[2i-1]
// //    编码得到部分积选择信号：0, ±1×A, ±2×A
// //    共产生 ceil(LEN/2) 个部分积，用Wallace树加法器压缩后得到最终结果。
// //
// //  结构：
// //    1. 输入寄存器（与原版相同，regenable 控制）
// //    2. Booth 编码器：对每组3位产生 (neg, two, one) 控制信号
// //    3. 部分积生成：根据控制信号选择 0 / A / 2A，neg时取反+修正
// //    4. 部分积压缩：使用进位保留加法器（CSA）Wallace树
// //    5. 最终加法：一个宽加法器得到结果
// // =============================================================================

// module naivemultiplier #(
//         parameter LEN = 4   // 操作数位宽，支持任意正偶数
//     )(
//         input              clk      ,
//         input              rst_n    ,
//         input              regenable,
//         input  [LEN-1:0]   a        ,
//         input  [LEN-1:0]   b        ,
//         output [LEN*2-1:0] result
//     );

//     // -------------------------------------------------------------------------
//     // 输入寄存器（与原版相同）
//     // -------------------------------------------------------------------------
//     reg [LEN-1:0] reg_a;
//     reg [LEN-1:0] reg_b;

//     always @(posedge clk or negedge rst_n) begin
//         if (!rst_n) begin
//             reg_a <= 'd0;
//             reg_b <= 'd0;
//         end else if (regenable) begin
//             reg_a <= a;
//             reg_b <= b;
//         end
//     end

//     // -------------------------------------------------------------------------
//     // Radix-4 Booth 编码
//     //
//     // 部分积个数 NUM_PP = ceil(LEN/2)
//     // 对 reg_b 补0扩展后每次取3位：
//     //   group i = { reg_b[2i+1], reg_b[2i], reg_b[2i-1] }，b[-1]=0
//     //
//     // 编码表：
//     //   b2i+1  b2i  b2i-1 | 操作
//     //   0      0    0     |  0
//     //   0      0    1     | +A
//     //   0      1    0     | +A
//     //   0      1    1     | +2A
//     //   1      0    0     | -2A
//     //   1      0    1     | -A
//     //   1      1    0     | -A
//     //   1      1    1     |  0
//     //
//     // 用三个信号表示：
//     //   neg : 取负（补码 = 取反 + 1，+1用符号位修正法处理）
//     //   two : 选 2A（左移一位）
//     //   one : 选 A（two=0时有效）
//     // -------------------------------------------------------------------------
//     localparam NUM_PP = (LEN + 1) / 2;  // ceil(LEN/2)

//     // 扩展 reg_b：在最低位前补一个0，在最高位后补一个0（保证最后一组有b[LEN]）
//     // 总共需要 LEN+1 位（b[LEN:0]），b[-1]=0 通过直接取 b_ext[0]=0 实现
//     wire [LEN+1:0] b_ext;
//     assign b_ext = {1'b0, reg_b, 1'b0};  // b_ext[i+1] = reg_b[i], b_ext[0]=0

//     // 每组 Booth 编码信号
//     wire [NUM_PP-1:0] booth_neg;
//     wire [NUM_PP-1:0] booth_two;
//     wire [NUM_PP-1:0] booth_one;

//     genvar i;
//     generate
//         for (i = 0; i < NUM_PP; i = i + 1) begin : booth_enc
//             // 取 b_ext[2i+2 : 2i]，对应原始 b[2i+1 : 2i-1]（b[-1]=0）
//             wire [2:0] grp = b_ext[2*i+2 : 2*i];

//             assign booth_neg[i] =  grp[2];                          // 最高位为1则取负
//             assign booth_two[i] = (grp[2] ^ grp[1]) & (grp[2] ^ grp[0]) ? 1'b0   // 排除±A
//                                 : (grp == 3'b100 || grp == 3'b011);                // ±2A
//             assign booth_one[i] = ~booth_two[i] & (grp[1] ^ grp[0]                // 仅 ±A
//                                    || (grp[2] & ~grp[1] & grp[0])
//                                    || (~grp[2] & grp[1] & ~grp[0]));
//         end
//     endgenerate

//     // 用更简洁的方式重新定义，直接查表
//     // neg = b2
//     // two = (b2 & ~b1 & ~b0) | (~b2 & b1 & b0)   即 011 或 100
//     // one = (b0 ^ b1) & ~(two)                     即非0非±2A时，b0≠b1则选A
//     // 重新generate（替换上面的enc）
//     wire [NUM_PP-1:0] enc_neg;
//     wire [NUM_PP-1:0] enc_two;
//     wire [NUM_PP-1:0] enc_one;

//     generate
//         for (i = 0; i < NUM_PP; i = i + 1) begin : booth_enc2
//             wire b_prev = b_ext[2*i  ];   // b[2i-1]，b[-1]=0
//             wire b_curr = b_ext[2*i+1];   // b[2i]
//             wire b_next = b_ext[2*i+2];   // b[2i+1]

//             assign enc_neg[i] =  b_next;
//             assign enc_two[i] = (b_next & ~b_curr & ~b_prev) | (~b_next & b_curr & b_prev);
//             assign enc_one[i] = b_curr ^ b_prev;   // one = b_curr XOR b_prev（two=0时才有效）
//         end
//     endgenerate

//     // -------------------------------------------------------------------------
//     // 部分积生成
//     //
//     // 部分积宽度：2*LEN+1 bits（留符号扩展空间）
//     // 每个部分积 PP[i] 左移 2*i 位对齐。
//     //
//     // 取反+1的修正：
//     //   neg时部分积取反，然后需要+1。
//     //   用"符号位修正法"：在最低有效位加一个1，等价于在 PP[i] 的 bit[2i] 加1。
//     //   将这个+1并入后续加法树（correction向量）。
//     //
//     // 部分积值：
//     //   enc_two=1, enc_neg=0 → +2A → reg_a << 1
//     //   enc_two=1, enc_neg=1 → -2A → ~(reg_a << 1) + 修正
//     //   enc_two=0, enc_one=1, enc_neg=0 → +A → reg_a
//     //   enc_two=0, enc_one=1, enc_neg=1 → -A → ~reg_a + 修正
//     //   其余 → 0
//     // -------------------------------------------------------------------------
//     localparam PP_WIDTH = 2*LEN + 2;  // 每个部分积的位宽（含符号扩展）

//     wire [PP_WIDTH-1:0] pp     [0:NUM_PP-1];  // 部分积（已移位对齐）
//     wire [NUM_PP-1:0]   pp_cor ;              // 各部分积的+1修正位

//     wire [LEN:0]   a_ext  = {reg_a[LEN-1], reg_a};  // 符号扩展1位，方便2A
//     wire [LEN+1:0] a2_ext = {reg_a[LEN-1], reg_a, 1'b0}; // 2A，符号扩展

//     generate
//         for (i = 0; i < NUM_PP; i = i + 1) begin : pp_gen
//             // 选择 |PP| 的值（未取反）
//             wire [LEN+1:0] pp_abs;
//             assign pp_abs = enc_two[i] ? a2_ext :          // 2A
//                             enc_one[i] ? {a_ext[LEN], a_ext} : // A（再扩1位对齐宽度）
//                                          'd0;               // 0

//             // 如果 neg，取反；否则保持
//             wire [LEN+1:0] pp_maybe_inv;
//             assign pp_maybe_inv = enc_neg[i] ? ~pp_abs : pp_abs;

//             // 符号扩展到 PP_WIDTH，并左移 2*i 位
//             // 用符号位填充高位（带符号扩展的移位）
//             wire [PP_WIDTH-1:0] pp_shifted;
//             assign pp_shifted = {{(PP_WIDTH-LEN-2){pp_maybe_inv[LEN+1]}},
//                                   pp_maybe_inv} << (2*i);

//             assign pp[i]     = pp_shifted;
//             assign pp_cor[i] = enc_neg[i]; // neg时在 bit[2i] 处需要+1
//         end
//     endgenerate

//     // 修正向量：将所有 pp_cor[i] 在对应位置合并成一个数
//     wire [PP_WIDTH-1:0] correction;
//     generate
//         // correction = sum of (pp_cor[i] << 2*i)
//         // 直接用一个宽加法拼接
//         wire [PP_WIDTH-1:0] cor_bits [0:NUM_PP-1];
//         for (i = 0; i < NUM_PP; i = i + 1) begin : cor_gen
//             assign cor_bits[i] = {{(PP_WIDTH-1){1'b0}}, pp_cor[i]} << (2*i);
//         end
//     endgenerate

//     // cor_bits 求和（NUM_PP 个单比特，用或叠加即可，因为每个在不同位）
//     // 实际上因为每个 cor 在不同的 bit 位置，直接 OR 即可
//     wire [PP_WIDTH-1:0] correction_sum;
//     generate
//         if (NUM_PP == 1) begin
//             assign correction_sum = cor_bits[0];
//         end else begin
//             wire [PP_WIDTH-1:0] cor_tmp [0:NUM_PP-1];
//             assign cor_tmp[0] = cor_bits[0];
//             for (i = 1; i < NUM_PP; i = i + 1) begin : cor_sum
//                 assign cor_tmp[i] = cor_tmp[i-1] | cor_bits[i];
//             end
//             assign correction_sum = cor_tmp[NUM_PP-1];
//         end
//     endgenerate

//     // -------------------------------------------------------------------------
//     // 部分积压缩（Wallace树 CSA）
//     //
//     // CSA（进位保留加法器）：3输入 → 2输出（sum, carry）
//     // 反复用 CSA 将部分积个数从 N 减少到 2，最后用一个加法器。
//     //
//     // 对于 LEN=4，NUM_PP=2，不需要 CSA，直接两个相加。
//     // 对于 LEN=8，NUM_PP=4，需要一级 CSA。
//     // 对于更大的 LEN，多级 CSA。
//     //
//     // 这里用参数化的方式：把所有 pp[] 和 correction 放入数组，
//     // 循环做 CSA 压缩直到剩2个，再做最终加法。
//     // -------------------------------------------------------------------------

//     // 把 NUM_PP 个部分积 + 1 个 correction 放入初始列表
//     // 总共 NUM_PP+1 个数
//     localparam TOTAL_PP = NUM_PP + 1;

//     // 用一个足够深的二维数组存放每一级的加数
//     // 最大级数：ceil(log1.5(TOTAL_PP)) 约为 TOTAL_PP
//     // 简单处理：直接展开 generate 做多级 CSA

//     // 将所有部分积收集到一个数组
//     wire [PP_WIDTH-1:0] adder_in [0:TOTAL_PP-1];
//     generate
//         for (i = 0; i < NUM_PP; i = i + 1) begin : pp_collect
//             assign adder_in[i] = pp[i];
//         end
//     endgenerate
//     assign adder_in[NUM_PP] = correction_sum;

//     // CSA 压缩：每次取前3个 → 产生 sum + carry，替换掉3个变2个
//     // 用递归 generate 参数化实现 Wallace 树
//     // 由于 Verilog generate 不支持真正递归，改用固定多级展开
//     // 对于 LEN<=32（NUM_PP<=16，TOTAL_PP<=17），5级 CSA 足够

//     // Level 0 输入：TOTAL_PP 个数
//     // 每级 CSA 数量 = floor(count/3)，剩余 = count mod 3 直接传下去
//     // Level 结束条件：剩余 <= 2

//     // 为简洁起见，用一个足够大的数组，逐级处理
//     // 最多支持 TOTAL_PP <= 3^5 = 243 个，对应 LEN <= 484，足够实用

//     localparam MAX_LEVELS = 6;
//     localparam MAX_TERMS  = TOTAL_PP; // 初始项数

//     // 手动展开 Wallace 树（参数化展开）
//     // 使用 function 计算每级项数
//     function integer csa_out_count;
//         input integer n;
//         begin
//             csa_out_count = (n/3)*2 + (n % 3);
//         end
//     endfunction

//     // 各级项数
//     localparam L0 = TOTAL_PP;
//     localparam L1 = (L0/3)*2 + (L0 % 3);
//     localparam L2 = (L1/3)*2 + (L1 % 3);
//     localparam L3 = (L2/3)*2 + (L2 % 3);
//     localparam L4 = (L3/3)*2 + (L3 % 3);
//     localparam L5 = (L4/3)*2 + (L4 % 3);

//     // 各级数组（取最大可能项数）
//     wire [PP_WIDTH-1:0] lv1 [0:L1-1];
//     wire [PP_WIDTH-1:0] lv2 [0:L2 > 1 ? L2-1 : 0];
//     wire [PP_WIDTH-1:0] lv3 [0:L3 > 1 ? L3-1 : 0];
//     wire [PP_WIDTH-1:0] lv4 [0:L4 > 1 ? L4-1 : 0];
//     wire [PP_WIDTH-1:0] lv5 [0:L5 > 1 ? L5-1 : 0];

//     // CSA 宏：对三个输入做进位保留加法
//     // sum[k]   = a[k] ^ b[k] ^ c[k]
//     // carry[k] = majority(a[k],b[k],c[k])，carry 左移1位后是下一级的加数

//     generate
//         // Level 0 → Level 1
//         for (i = 0; i < L0/3; i = i + 1) begin : csa_l0
//             wire [PP_WIDTH-1:0] csa_s, csa_c;
//             assign csa_s = adder_in[3*i] ^ adder_in[3*i+1] ^ adder_in[3*i+2];
//             assign csa_c = ((adder_in[3*i] & adder_in[3*i+1])
//                           | (adder_in[3*i+1] & adder_in[3*i+2])
//                           | (adder_in[3*i] & adder_in[3*i+2])) << 1;
//             assign lv1[2*i]   = csa_s;
//             assign lv1[2*i+1] = csa_c;
//         end
//         // 剩余项直接传
//         for (i = 0; i < L0 % 3; i = i + 1) begin : pass_l0
//             assign lv1[(L0/3)*2 + i] = adder_in[(L0/3)*3 + i];
//         end
//     endgenerate

//     generate
//         if (L1 > 2) begin : csa_lv1
//             for (i = 0; i < L1/3; i = i + 1) begin : csa_l1
//                 wire [PP_WIDTH-1:0] csa_s, csa_c;
//                 assign csa_s = lv1[3*i] ^ lv1[3*i+1] ^ lv1[3*i+2];
//                 assign csa_c = ((lv1[3*i] & lv1[3*i+1])
//                               | (lv1[3*i+1] & lv1[3*i+2])
//                               | (lv1[3*i] & lv1[3*i+2])) << 1;
//                 assign lv2[2*i]   = csa_s;
//                 assign lv2[2*i+1] = csa_c;
//             end
//             for (i = 0; i < L1 % 3; i = i + 1) begin : pass_l1
//                 assign lv2[(L1/3)*2 + i] = lv1[(L1/3)*3 + i];
//             end
//         end else begin : bypass_lv1
//             for (i = 0; i < L1; i = i + 1) begin : pass_lv1
//                 assign lv2[i] = lv1[i];
//             end
//         end
//     endgenerate

//     generate
//         if (L2 > 2) begin : csa_lv2
//             for (i = 0; i < L2/3; i = i + 1) begin : csa_l2
//                 wire [PP_WIDTH-1:0] csa_s, csa_c;
//                 assign csa_s = lv2[3*i] ^ lv2[3*i+1] ^ lv2[3*i+2];
//                 assign csa_c = ((lv2[3*i] & lv2[3*i+1])
//                               | (lv2[3*i+1] & lv2[3*i+2])
//                               | (lv2[3*i] & lv2[3*i+2])) << 1;
//                 assign lv3[2*i]   = csa_s;
//                 assign lv3[2*i+1] = csa_c;
//             end
//             for (i = 0; i < L2 % 3; i = i + 1) begin : pass_l2
//                 assign lv3[(L2/3)*2 + i] = lv2[(L2/3)*3 + i];
//             end
//         end else begin : bypass_lv2
//             for (i = 0; i < L2; i = i + 1) begin : pass_lv2
//                 assign lv3[i] = lv2[i];
//             end
//         end
//     endgenerate

//     generate
//         if (L3 > 2) begin : csa_lv3
//             for (i = 0; i < L3/3; i = i + 1) begin : csa_l3
//                 wire [PP_WIDTH-1:0] csa_s, csa_c;
//                 assign csa_s = lv3[3*i] ^ lv3[3*i+1] ^ lv3[3*i+2];
//                 assign csa_c = ((lv3[3*i] & lv3[3*i+1])
//                               | (lv3[3*i+1] & lv3[3*i+2])
//                               | (lv3[3*i] & lv3[3*i+2])) << 1;
//                 assign lv4[2*i]   = csa_s;
//                 assign lv4[2*i+1] = csa_c;
//             end
//             for (i = 0; i < L3 % 3; i = i + 1) begin : pass_l3
//                 assign lv4[(L3/3)*2 + i] = lv3[(L3/3)*3 + i];
//             end
//         end else begin : bypass_lv3
//             for (i = 0; i < L3; i = i + 1) begin : pass_lv3
//                 assign lv4[i] = lv3[i];
//             end
//         end
//     endgenerate

//     generate
//         if (L4 > 2) begin : csa_lv4
//             for (i = 0; i < L4/3; i = i + 1) begin : csa_l4
//                 wire [PP_WIDTH-1:0] csa_s, csa_c;
//                 assign csa_s = lv4[3*i] ^ lv4[3*i+1] ^ lv4[3*i+2];
//                 assign csa_c = ((lv4[3*i] & lv4[3*i+1])
//                               | (lv4[3*i+1] & lv4[3*i+2])
//                               | (lv4[3*i] & lv4[3*i+2])) << 1;
//                 assign lv5[2*i]   = csa_s;
//                 assign lv5[2*i+1] = csa_c;
//             end
//             for (i = 0; i < L4 % 3; i = i + 1) begin : pass_l4
//                 assign lv5[(L4/3)*2 + i] = lv4[(L4/3)*3 + i];
//             end
//         end else begin : bypass_lv4
//             for (i = 0; i < L4; i = i + 1) begin : pass_lv4
//                 assign lv5[i] = lv4[i];
//             end
//         end
//     endgenerate

//     // -------------------------------------------------------------------------
//     // 最终加法：取最后两个加数相加
//     // -------------------------------------------------------------------------
//     wire [PP_WIDTH-1:0] final_sum;
//     wire [PP_WIDTH-1:0] final_a;
//     wire [PP_WIDTH-1:0] final_b;

//     // 选取最后一级输出的前两项
//     assign final_a = lv5[0];
//     assign final_b = (L5 >= 2) ? lv5[1] : 'd0;

//     assign final_sum = final_a + final_b;

//     // -------------------------------------------------------------------------
//     // 输出：取低 2*LEN 位
//     // -------------------------------------------------------------------------
//     assign result = final_sum[2*LEN-1:0];

// endmodule