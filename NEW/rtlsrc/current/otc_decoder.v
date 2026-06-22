 `timescale 1ns/1ps

 module otc_decoder (
	input	[31:0]		instr_i,

	output	reg			dec_valid_o,
	output	reg			dec_is_tcu_o,
	output	reg			dec_illegal_o,
	
	output	reg			dec_is_load_o,
	output	reg			dec_is_store_o,
	output	reg			dec_is_wmma_o,
	output	reg			dec_is_wmma_sp_o,

	output	reg	[4:0]		dec_rd_o,
	output	reg	[4:0]	    dec_rs1_o,
    output	reg	[4:0]       dec_rs2_o,

    output	reg	[2:0]       dec_shape_o,
    output	reg	[2:0]       dec_num_o,
    output	reg	            dec_pack_o,
    output	reg	            dec_unpack_o,

    output	reg	[2:0]       dec_type_o,
    output	reg	            dec_acc_o,
    output	reg	            dec_at_o,
    output	reg	            dec_bt_o,
    output	reg	[1:0]       dec_spsel_o,
    output	reg	[2:0]       dec_ctx_id_o
    
	         
);

wire       [6:0]           opcode;
wire       [4:0]           rd;
wire       [2:0]           funct3;
wire       [4:0]           rs1;
wire       [4:0]           rs2;
wire       [6:0]           funct7;

assign      opcode      =   instr_i[6:0];
assign      rd          =   instr_i[11:7];
assign      funct3      =   instr_i[14:12];
assign      rs1         =   instr_i[19:15];
assign      rs2         =   instr_i[24:20];
assign      funct7      =   instr_i[31:25];

parameter       OPC_TCU_LOAD     =   7'b0001011;
parameter       OPC_TCU_STORE    =   7'b0101011;
parameter       OPC_TCU_WMMA     =   7'b1011011;

parameter       F3_WMMA_DENSE     =   3'b000;
parameter       F3_WMMA_SPARSE    =   3'b001;



always @(*)    begin
    dec_valid_o         =   1'b0;
    dec_is_tcu_o        =   1'b0;
    dec_illegal_o       =   1'b0;

    dec_is_load_o       =   1'b0;
    dec_is_store_o      =   1'b0;
    dec_is_wmma_o       =   1'b0;
    dec_is_wmma_sp_o    =   1'b0;

    dec_rd_o            =   rd;
    dec_rs1_o           =   rs1;
    dec_rs2_o           =   rs2;

    dec_shape_o         =   3'b000;
    dec_num_o           =   3'b000;
    dec_pack_o          =   1'b0;
    dec_unpack_o        =   1'b0;

    dec_type_o          =   3'b000;
    dec_acc_o           =   1'b0;
    dec_at_o            =   1'b0;
    dec_bt_o            =   1'b0;
    dec_spsel_o         =   2'b00;
    dec_ctx_id_o        =   3'b000;

    case    (opcode)

        OPC_TCU_LOAD:begin
            dec_is_tcu_o        =   1'b1;
            dec_valid_o         =   1'b1;
            dec_is_load_o       =   1'b1;

            dec_shape_o         =   funct7[6:4];
            dec_num_o           =   funct7[3:1];
            dec_pack_o          =   funct7[0];

            if(funct3 != 3'b000 || rs2 != 5'b00000) begin
              dec_illegal_o     =   1'b1;
              dec_valid_o       =   1'b0;
            end
        end 


        OPC_TCU_STORE:begin
            dec_is_tcu_o        =   1'b1;
            dec_valid_o         =   1'b1;
            dec_is_store_o       =   1'b1;

            dec_shape_o         =   funct7[6:4];
            dec_num_o           =   funct7[3:1];
            dec_unpack_o        =   funct7[0];

            if(funct3 != 3'b000 || rs2 != 5'b00000) begin
              dec_illegal_o     =   1'b1;
              dec_valid_o       =   1'b0;
            end
        end



        OPC_TCU_WMMA:begin
            dec_is_tcu_o        =   1'b1;
            dec_ctx_id_o        =   rd[2:0];
            case (funct3)
                F3_WMMA_DENSE:begin
                    if(funct7[6] == 1'b0) begin
                        dec_valid_o         =   1'b1;
                        dec_is_wmma_o       =   1'b1;

                        dec_type_o          =   funct7[5:3];
                        dec_acc_o           =   funct7[2];
                        dec_at_o            =   funct7[1];
                        dec_bt_o            =   funct7[0];

                        if(rd[4:3] != 2'b00) begin
                            dec_illegal_o       =   1'b1;
                            dec_valid_o         =   1'b0;
                        end
                    end
                    else begin
                      dec_illegal_o     =   1'b1;
                    end
                    end 

                    F3_WMMA_SPARSE:begin
                        if(funct7[6] == 1'b1) begin
                        dec_valid_o         =   1'b1;
                        dec_is_wmma_sp_o       =   1'b1;

                        dec_type_o          =   funct7[5:3];
                        dec_acc_o           =   funct7[2];
                        dec_spsel_o         =   funct7[1:0];

                        if(rd[4:3] != 2'b00) begin
                            dec_illegal_o       =   1'b1;
                            dec_valid_o         =   1'b0;
                        end
                    end
                    else begin
                      dec_illegal_o     =   1'b1;
                    end
                    end 

                    default:begin
                        dec_illegal_o       =   1'b1;
                    end
            endcase
        end

        default:begin
            dec_valid_o         =   1'b0;
            dec_is_tcu_o        =   1'b0;
            dec_illegal_o       =   1'b0;
        end
    endcase
end
endmodule