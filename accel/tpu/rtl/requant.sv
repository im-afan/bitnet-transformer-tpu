module requant #(
    parameter integer QUANT_W = 8,
    parameter integer M0_W = 12,
    parameter integer N_W = 4
) (
    input logic clk,
    input logic rst_n,

    input logic [QUANT_W-1:0] acc,
    input logic [M0_W-1:0] m0,
    input logic [N_W-1:0] n,
    output logic [QUANT_W-1:0] result
);
    // logic [QUANT_W+M0_W-1 : 0] wide = acc * m0 + (1 << (n-1));
    // assign result = wide >> n;
endmodule
