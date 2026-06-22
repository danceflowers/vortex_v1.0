module csa3_2 #(
    parameter WIDTH = 32
)(
    input  [WIDTH-1:0] a,
    input  [WIDTH-1:0] b,
    input  [WIDTH-1:0] c,
    output [WIDTH-1:0] sum,
    output [WIDTH-1:0] carry
);
    assign sum   = a ^ b ^ c;
    assign carry = (a & b) | (a & c) | (b & c);
endmodule
