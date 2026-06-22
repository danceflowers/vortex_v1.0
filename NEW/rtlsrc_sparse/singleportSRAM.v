module singleportSRAM #(
    parameter D_WIDTH=24, // the data width depends on the expected data type which is specified by instruction 
    parameter A_WIDTH=4
)(
    input clk,
    input rst_n,   
    //input w_data_i,   
    input r_ready_i,  
    input w_valid_i,
      
   // input r_data_o,   
    //input r_valid_o,  
	//input w_ready_o,
	output w_ready_o, 
    output r_valid_o,
    // input csen_n, // may not use
    input w_en_i,
    input [A_WIDTH-1:0] addr,

    input [D_WIDTH-1:0] w_data_i,//fan fu ding yi
    output reg [D_WIDTH-1:0] r_data_o//fan fu ding yi
);
	assign r_valid_o = rst_n ? (!w_en_i) : 1'b0;
	assign w_ready_o = 1'b1;

    reg [D_WIDTH-1:0] mem [2**A_WIDTH-1:0];
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            for (integer i=0; i<2**A_WIDTH; i=i+1)
                mem[i]<={D_WIDTH{1'b0}};
        end
        else if(w_en_i)
            mem[addr]<=w_data_i;
    end
    always @(posedge clk) begin
        if(!w_en_i)
        r_data_o <= w_data_i;   
		// r_data_o<=mem[addr];
        else
        //    r_data_o<={D_WIDTH{1'bz}};
			r_data_o<=mem[addr];
    end
endmodule
