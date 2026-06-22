module to_next_con #(
    parameter BANK_NUM   = 8,
    parameter BANK_WIDTH = 64,
    parameter DEPTH      = 16,
    parameter BUS_WIDTH  = 512,
    parameter OUT_WIDTH  = 1408
)(
    input clk,
    input rst_n,

    input [1:0] dtype,  
    input       rd_start_i,

    // input stream
    input  [BUS_WIDTH-1:0] in_data,
    input                  in_valid_i,
    output                 in_ready_o,  //读ready

    input out_ready_i,

    // output
    output [OUT_WIDTH-1:0] out_data,
    output          out_valid_o
);

    // ============================================================
    // SRAM
    // ============================================================
    reg [1:0] wr_en;
    reg [$clog2(DEPTH)-1:0] wr_addr;

    wire [BANK_NUM*BANK_WIDTH-1:0] rd_data;
    wire rd_valid;

    reg [1:0] rd_en;
    reg [$clog2(DEPTH)-1:0] rd_addr;

    banked_sram #(
        .BANK_NUM(BANK_NUM),
        .BANK_WIDTH(BANK_WIDTH),
        .DEPTH(DEPTH)
    ) u_sram (
        .clk(clk),
        .rst_n(rst_n),

        .wr_en(wr_en),
        .wr_addr(wr_addr),
        .wr_data(in_data),

        .rd_en(rd_en),
        .rd_addr(rd_addr),

        .rd_data(rd_data),
        .rd_valid(rd_valid)
    );

    // ============================================================
    // 写控制（简单顺序写）
    // ============================================================

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            wr_en   <= 0;
        end else begin
            if(in_valid_i) begin
                wr_en   <= 2'b11;
            end else begin
                wr_en <= 2'b00;
            end
        end
    end
    always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        wr_addr <= 0;
    end else begin
        if(wr_en==2'b11) begin
            wr_addr <= wr_addr + 1;
        end else begin
            wr_addr <= wr_addr;
        end
    end
end

    // ============================================================
    // dtype decode
    // ============================================================
    reg [2:0] total_cycles;
    reg [6:0] elems_per_cycle;

    always @(*) begin
        case(dtype)
            2'b01,2'b00: begin total_cycles=1; elems_per_cycle=64; end
            2'b10:       begin total_cycles=2; elems_per_cycle=32; end
            2'b11:       begin total_cycles=4; elems_per_cycle=16; end
            default:     begin total_cycles=1; elems_per_cycle=64; end
        endcase
    end

    // ============================================================
    // read FSM
    // ============================================================
    localparam IDLE=0, RUN=1;

    reg state;
    reg [2:0] cycle_cnt;

    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) state <= IDLE;
        else begin
            case(state)
                IDLE: if(rd_start_i) state <= RUN;
                RUN:  if(rd_valid && cycle_cnt==total_cycles-1) state <= IDLE;
            endcase
        end
    end

    // read control
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            rd_en   <= 0;
            rd_addr <= 0;
            cycle_cnt <= 0;
        end else begin
            case(state)

            IDLE: begin
                if(rd_start_i) begin    //todo:为了在不同的数据精度中对齐读到的C数据，可能需要在fp8时将这个信号延后3个周期，在fp16时延后两个周期
                    rd_en   <= 2'b11;
                    rd_addr <= rd_addr;   //todo:
                    cycle_cnt <= 0;
                end
            end

            RUN: begin
                if(rd_en==2'b11) begin
                    rd_addr   <= rd_addr + 1;
                    end

               if(cycle_cnt == total_cycles-1) begin
                        rd_en <= 0;      end
                
                if(cycle_cnt == total_cycles-1)begin
                    cycle_cnt <=0 ; end
                // else if(rd_en==2'b11)
                 //   cycle_cnt <= cycle_cnt + 1;
                else 
                    cycle_cnt <= cycle_cnt + 1;

 
      
            end

            endcase
        end
    end

    // ============================================================
    // 512bit extract
    // ============================================================
    wire [511:0] rd_data_512 = rd_data[511:0];

    // ============================================================
    // FP转换（每拍）
    // ============================================================
    wire [1407:0] fp22_partial;

    fp_to_fp22 u_fp22 (
        .clk(clk),
        .rst_n(rst_n),

        .in_i(rd_data_512),
        .dtype_i(dtype),

        .in_valid_i(rd_valid),
        .out_ready_i(1'b1),

        .out_o(fp22_partial),
        .out_valid_o(),
        .in_ready_o()
    );

    // ============================================================
    // FP22 buffer（64 lanes）
    // ============================================================
    reg [21:0] fp22_buf [0:63];

    integer i;

    always @(posedge clk) begin
        if(rd_valid) begin
            // for(i=0;i<64;i=i+1) begin
            //     if(i < elems_per_cycle) begin
            //         fp22_buf[cycle_cnt*elems_per_cycle + i]
            //             <= fp22_partial[i*22 +: 22];
            for(i = 0; i < elems_per_cycle; i = i + 1) begin
                fp22_buf[cycle_cnt*elems_per_cycle + i]
                    <= fp22_partial[i*22 +: 22];
                    end
                end
        end

    // ============================================================
    // 输出拼接
    // ============================================================
    reg [1407:0] out_r;
    integer j;
    always @(*) begin
        for(j=0;j<64;j=j+1)
            out_r[j*22 +: 22] = fp22_buf[j];
    end
    assign out_data = out_r;

    // ============================================================
    // valid
    // ============================================================
    //数据转换有一个周期，所以rd_valid和cycle_cnt判断转换后的数据有效时，需要打一拍
    reg rd_valid_r;
    reg [2:0] cycle_cnt_r;
    reg out_valid_r;

    always @(posedge clk) begin
        rd_valid_r <= rd_valid;
        cycle_cnt_r <= cycle_cnt;
    end
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n)
            out_valid_r <= 0;
        else
            out_valid_r <= (rd_valid && cycle_cnt_r==total_cycles-1);
    end
    assign in_ready_o = state==IDLE;  //只要不在读状态就ready
    assign out_valid_o = out_valid_r;

endmodule